"""Apply trained models to committed daily reanalysis as an external job."""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .precip import PrecipCorrector
from .temperature import TemperatureCorrector
from weatheasy.storage import delete_group, last_date, open_dataset, prepare_dataset, write_batch


if TYPE_CHECKING:
    import zarr
    from zarr.abc.store import Store

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)
_FIELDS = {'TMP_corr', 'PRATE_corr'}


def correct_store(  # noqa: C901, PLR0912, PLR0915
    source: Source,
    root: zarr.Group,
    artifacts: FileStore,
    dem_store: Store,
    *,
    begin: date | None = None,
    end: date | None = None,
) -> int:
    """Append corrected days; each completed day contains both corrected fields.

    A changed or partial spatial correction group is rebuilt from the first raw
    day. No independent corr_last is needed: committed_length is the frontier.
    """
    with open_dataset(root.store, source.path(root)) as raw:
        raw_dates = [
            date.fromisoformat(str(value)) for value in raw.time.to_numpy().astype('datetime64[D]')
        ]
        crs = raw.rio.crs
    raw_first, raw_last = raw_dates[0], raw_dates[-1]
    if begin is not None and end is not None and begin > end:
        msg = 'Correction begin must not follow end'
        raise ValueError(msg)
    if end is not None and end > raw_last:
        msg = f'Correction ends after the last committed {source.name} day {raw_last}'
        raise ValueError(msg)
    if begin is not None and begin < raw_first:
        msg = f'Correction begins before the first committed {source.name} day {raw_first}'
        raise ValueError(msg)

    lats, lons = source.grid(root)
    temp = TemperatureCorrector(source, artifacts, dem_store).load_artifacts()
    precip = PrecipCorrector(source, artifacts, dem_store).load_artifacts()
    temp_climate = temp.climatology
    precip_climate = precip.climatology
    if temp_climate is not None and precip_climate is not None:
        temp_climate.require_grid(lats, lons)
        precip_climate.require_grid(lats, lons)
        for key in ('first_day', 'last_day', 'day_count', 'days_sha256'):
            if temp_climate.metadata[key] != precip_climate.metadata[key]:
                msg = (
                    f'{source.name}: temperature and precipitation artifacts use '
                    'different raw snapshots; retrain both'
                )
                raise ValueError(msg)
    ids = f'{temp.artifact_id}:{precip.artifact_id}'
    artifact_id = hashlib.sha256(ids.encode()).hexdigest()
    path = source.corrected_path(root)
    if _requires_rebuild(root, source.correction_path, lats, lons, artifact_id):
        _LOG.warning(
            '%s correction grid is incompatible; rebuilding from %s',
            source.name,
            raw_first,
        )
        delete_group(root.store, path)
        frontier = None
        begin = raw_first  # Old progress no longer applies to the recreated arrays.
    else:
        frontier = last_date(root.store, path)

    if frontier is None:
        pending = [
            day
            for day in raw_dates
            if (begin is None or day >= begin) and (end is None or day <= end)
        ]
    else:
        pending = [day for day in raw_dates if day > frontier and (end is None or day <= end)]
        if begin is not None and pending and begin > pending[0]:
            msg = f'Correction cannot skip prepared day {pending[0]} after {frontier}'
            raise ValueError(msg)
    if not pending:
        return 0

    if crs is None:
        msg = f'{source.name} prepared data have no CRS'
        raise ValueError(msg)
    with open_dataset(root.store, source.path(root)) as raw:
        view = source._crop(raw)  # noqa: SLF001
        source_group = root.get_group(source.group_path)
        chunk_size = source_group.get_array(source.temp_var).chunks[0]
        lat_offset = int(np.searchsorted(raw.latitude.to_numpy(), view.latitude.to_numpy()[0]))
        lon_offset = int(np.searchsorted(raw.longitude.to_numpy(), view.longitude.to_numpy()[0]))
        indices = np.flatnonzero(
            np.isin(
                view.time.to_numpy().astype('datetime64[D]'),
                np.array(pending, dtype='datetime64[D]'),
            )
        )
        for chunk_index in np.unique(indices // chunk_size):
            selected = indices[indices // chunk_size == chunk_index]
            with TemporaryDirectory(prefix='weatheasy-correction-') as directory:
                block = view[[source.temp_var, source.precip_var]].isel(time=selected)
                shape = (len(selected), len(lats), len(lons))
                inputs = {}
                outputs = {}
                for var in (source.temp_var, source.precip_var):
                    inputs[var] = np.memmap(
                        Path(directory) / f'{var}.raw', mode='w+', dtype=np.float32, shape=shape
                    )
                    _, lat_step, lon_step = source_group.get_array(var).chunks
                    lat_edges = [
                        0,
                        *range(lat_step - lat_offset % lat_step, len(lats), lat_step),
                        len(lats),
                    ]
                    lon_edges = [
                        0,
                        *range(lon_step - lon_offset % lon_step, len(lons), lon_step),
                        len(lons),
                    ]
                    for lat_start, lat_stop in pairwise(lat_edges):
                        for lon_start, lon_stop in pairwise(lon_edges):
                            spatial = (slice(lat_start, lat_stop), slice(lon_start, lon_stop))
                            inputs[var][(slice(None), *spatial)] = (
                                block[var]
                                .isel(latitude=spatial[0], longitude=spatial[1])
                                .to_numpy()
                            )
                for var in ('TMP_corr', 'PRATE_corr'):
                    outputs[var] = np.memmap(
                        Path(directory) / f'{var}.raw', mode='w+', dtype=np.float32, shape=shape
                    )
                for index, value in enumerate(block.time.to_numpy().astype('datetime64[D]')):
                    day = date.fromisoformat(str(value))
                    outputs['TMP_corr'][index] = temp.correct_grid(
                        day, inputs[source.temp_var][index], lats, lons
                    )['corrected']
                    outputs['PRATE_corr'][index] = precip.correct_grid(
                        day, inputs[source.precip_var][index], lats, lons
                    )['corrected']
                ds = xr.Dataset(
                    {
                        'TMP_corr': (
                            ('time', 'latitude', 'longitude'),
                            outputs['TMP_corr'],
                            {'units': 'degC', 'long_name': 'Bias-corrected daily temperature'},
                        ),
                        'PRATE_corr': (
                            ('time', 'latitude', 'longitude'),
                            outputs['PRATE_corr'],
                            {'units': 'mm/day', 'long_name': 'Bias-corrected daily precipitation'},
                        ),
                    },
                    coords={'time': block.time, 'latitude': lats, 'longitude': lons},
                )
                prepared = prepare_dataset(ds, crs=crs)
                prepared.attrs['correction_artifact_id'] = artifact_id
                write_batch(root.store, path, prepared)
                del inputs, outputs, ds, prepared
    return len(pending)


def _requires_rebuild(
    root: zarr.Group,
    group_path: str,
    lats: np.ndarray,
    lons: np.ndarray,
    artifact_id: str,
) -> bool:
    if group_path not in root:
        return False
    group = cast('zarr.Group', root[group_path])
    if not group.attrs or not any(group.keys()):
        return False
    if group.attrs.get('correction_artifact_id') != artifact_id:
        return True
    # Inspect even an interrupted append: its published coordinate arrays stay
    # unchanged, and write_batch can recover the uncommitted time tail.
    try:
        with xr.open_zarr(
            root.store,
            group='/'.join(filter(None, (root.path, group_path))),
            consolidated=False,
            chunks=None,
            decode_coords='all',
            zarr_format=3,
        ) as ds:
            return (
                set(ds.data_vars) != _FIELDS
                or not np.array_equal(ds.latitude, lats)
                or not np.array_equal(ds.longitude, lons)
            )
    except (KeyError, ValueError):
        return True
