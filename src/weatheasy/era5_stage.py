"""Restartable, bounded local storage for daily ERA5 reductions."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from weatheasy.storage import StorageError, _signature, write_staged


if TYPE_CHECKING:
    from pathlib import Path

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

    def append(self, day: xr.Dataset) -> None:
        if day.sizes['time'] != 1:
            msg = 'ERA5 staging requires one calendar day at a time'
            raise StorageError(msg)
        if not self.written:
            self.template = day.isel(time=slice(0, 0)).copy()
        elif _signature(day) != _signature(self.template):
            msg = 'ERA5 staging has a conflicting grid, CRS, variables or units'
            raise StorageError(msg)
        times = day.time.to_numpy().astype('datetime64[s]').astype(np.int64)
        if self.written:
            previous = np.fromfile(
                self.filenames['time'], dtype=np.int64, count=1, offset=(self.written - 1) * 8
            )[0]
            if times[0] != previous + 86400:
                msg = 'ERA5 staged days must be consecutive'
                raise StorageError(msg)
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
