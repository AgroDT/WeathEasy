"""Restartable, bounded local storage for daily ERA5 reductions."""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from zarr.storage import LocalStore

from weatheasy import const
from weatheasy.raw_cache import default_work_dir
from weatheasy.storage import StorageError, _signature, group_attrs, set_group_attrs, write_staged


if TYPE_CHECKING:
    from zarr.abc.store import Store


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    msg = f'Unsupported ERA5 metadata type: {type(value).__name__}'
    raise TypeError(msg)


class DailyStage:
    """Append raw fields before atomically advancing the daily checkpoint.

    Bytes beyond the checkpoint are overwritten on retry. Publishing never
    removes local data: acknowledge only after the remote commit succeeds.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.manifest = directory / 'state.json'
        self.written = 0
        self.template = xr.Dataset()
        if self.manifest.exists():
            state = json.loads(self.manifest.read_text())
            self.written = state['written']
            self.template = xr.Dataset.from_dict(state['coordinates'])
            self.template = self.template.assign_coords(
                time=xr.DataArray(
                    np.array([], dtype='datetime64[ns]'),
                    dims='time',
                    attrs=self.template.time.attrs,
                )
            )
            for name, field in state['fields'].items():
                self.template[name] = xr.DataArray(
                    np.empty(
                        (0, self.template.sizes['latitude'], self.template.sizes['longitude']),
                        dtype=field['dtype'],
                    ),
                    dims=('time', 'latitude', 'longitude'),
                    attrs=field['attrs'],
                )
            self.template.attrs = state['attrs']

    @property
    def filenames(self) -> dict[str, Path]:
        names = ['time', *sorted(str(name) for name in self.template.data_vars)]
        return {str(name): self.directory / f'{index}.raw' for index, name in enumerate(names)}

    def dates(self) -> tuple[date, date]:
        times = np.fromfile(self.filenames['time'], dtype=np.int64, count=self.written)
        if len(times) != self.written:
            msg = 'Truncated ERA5 staging timestamps'
            raise StorageError(msg)
        begin, end = (
            date.fromisoformat(str(np.datetime64(int(value), 's').astype('datetime64[D]')))
            for value in (times[0], times[-1])
        )
        return begin, end

    def check_compatible(self, day: xr.Dataset) -> None:
        """Check the first reduced day from new monthly inputs against retained staging."""
        if self.written and _signature(day) != _signature(self.template):
            msg = 'ERA5 staging has a conflicting grid, CRS, variables or units'
            raise StorageError(msg)

    def append(self, day: xr.Dataset) -> None:
        """Checkpoint one day from the caller's consecutive, compatible monthly batch."""
        if not self.written:
            self.template = day.isel(time=slice(0, 0)).copy()
        times = day.time.to_numpy().astype('datetime64[s]').astype(np.int64)
        for name, filename in self.filenames.items():
            values = times if name == 'time' else day[name].to_numpy()
            with filename.open('r+b' if filename.exists() else 'w+b') as stream:
                stream.seek(self.written * values.nbytes)
                values.tofile(stream)
                stream.truncate()
        state = {
            'written': self.written + 1,
            'coordinates': self.template.coords.to_dataset().to_dict(data='list'),
            'fields': {
                name: {'dtype': str(field.dtype), 'attrs': field.attrs}
                for name, field in self.template.data_vars.items()
            },
            'attrs': self.template.attrs,
        }
        pending = self.manifest.with_suffix('.part')
        pending.write_text(json.dumps(state, default=_json_scalar))
        pending.replace(self.manifest)
        self.written += 1

    def publish(self, store: Store) -> None:
        write_staged(store, 'daily', self.template, self.filenames, self.written, time_chunk=128)

    def clear(self) -> None:
        self.manifest.unlink(missing_ok=True)
        self.manifest.with_suffix('.part').unlink(missing_ok=True)
        for filename in self.filenames.values():
            filename.unlink(missing_ok=True)
        self.written = 0
        self.template = xr.Dataset()


LOGGER = logging.getLogger(__name__)


def resume_stage(
    store: Store, bbox: tuple, work_dir: str | Path | None, previous: date | None
) -> tuple[DailyStage, date, int]:
    first = previous + const.ONE_DAY if previous else date(const.ERA5_FIRST_YEAR, 1, 1)
    directory = Path(work_dir).expanduser() if work_dir is not None else default_work_dir()
    filesystem = getattr(store, 'fs', None)
    identity = json.dumps(
        {
            'store': str(store.root.resolve()) if isinstance(store, LocalStore) else str(store),
            'bbox': [float(value) for value in bbox],
            'endpoint': getattr(filesystem, 'client_kwargs', {}).get('endpoint_url')
            or os.environ.get('AWS_ENDPOINT_URL_S3')
            or os.environ.get('AWS_ENDPOINT_URL'),
        },
        sort_keys=True,
    )
    namespace = hashlib.sha256(identity.encode()).hexdigest()[:20]
    stage = DailyStage(directory / 'era5' / 'staging' / namespace)
    if stage.written:
        begin, end = stage.dates()
        if previous and end <= previous:
            # The remote commit succeeded before local acknowledgement.
            stage.clear()
        elif begin != first:
            msg = 'ERA5 staging does not follow the committed period'
            raise StorageError(msg)
        else:
            first = end + const.ONE_DAY
    count = int(group_attrs(store, 'daily')['committed_length']) if previous else 0
    capacity = 128 - count % 128
    if stage.written > capacity:
        msg = 'ERA5 staging exceeds the next time chunk boundary'
        raise StorageError(msg)
    if stage.written == capacity:
        publish_stage(stage, store)
        count += capacity
    return stage, first, count


def month_marker(day: date) -> list[int]:
    last = (
        day
        if day.day == calendar.monthrange(day.year, day.month)[1]
        else (day.replace(day=1) - const.ONE_DAY)
    )
    return [last.year, last.month]


def publish_stage(stage: DailyStage, store: Store) -> None:
    if not stage.written:
        return
    begin, end = stage.dates()
    started = time.monotonic()
    LOGGER.info('Writing ERA5-Land block: %s through %s (%s days)', begin, end, stage.written)
    stage.publish(store)
    set_group_attrs(store, '', last_month=month_marker(end))
    stage.clear()
    LOGGER.info('Published ERA5-Land through %s in %.1f s', end, time.monotonic() - started)
