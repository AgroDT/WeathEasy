"""Apply trained models to committed daily reanalysis as an external job."""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import date
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .precip import PrecipCorrector
from .sources import CFS2_FORECAST
from .temperature import TemperatureCorrector
from weatheasy import const
from weatheasy.storage import (
    delete_group,
    group_attrs,
    group_exists,
    last_date,
    open_dataset,
    prepare_dataset,
    set_group_attrs,
    write_batch,
    write_batches,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence

    import zarr
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)
_FIELDS = {'TMP_corr', 'PRATE_corr'}
_FORECAST_BLOCK = 32  # days per forecast write; half of the forecast time chunk
_REGRID_MARGIN = 2.0  # degrees around the crop, wider than one T126 cell


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
    temp, precip, artifact_id = load_correctors(source, artifacts, dem_store, lats, lons)
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
                ds = _corrected_dataset(
                    block.time, lats, lons, outputs['TMP_corr'], outputs['PRATE_corr']
                )
                prepared = prepare_dataset(ds, crs=crs)
                prepared.attrs['correction_artifact_id'] = artifact_id
                write_batch(root.store, path, prepared)
                del inputs, outputs, ds, prepared
    return len(pending)


def load_correctors(
    source: Source,
    artifacts: FileStore,
    dem_store: Store,
    lats: NDArray[np.float32] | None = None,
    lons: NDArray[np.float32] | None = None,
) -> tuple[TemperatureCorrector, PrecipCorrector, str]:
    """Load a consistent temperature/precipitation model pair and its combined id.

    Without ``lats``/``lons`` the pair must share a grid, as for the forecast,
    which is interpolated onto the models' own grid.
    """
    temp = TemperatureCorrector(source, artifacts, dem_store).load_artifacts()
    precip = PrecipCorrector(source, artifacts, dem_store).load_artifacts()
    temp_climate = temp.climatology
    precip_climate = precip.climatology
    if temp_climate is not None and precip_climate is not None:
        if lats is None or lons is None:
            lats, lons = temp_climate.lats, temp_climate.lons
        temp_climate.require_grid(lats, lons)
        precip_climate.require_grid(lats, lons)
        for key in ('first_day', 'last_day', 'day_count', 'days_sha256'):
            if temp_climate.metadata[key] != precip_climate.metadata[key]:
                msg = (
                    f'{source.model.name}: temperature and precipitation artifacts use '
                    'different raw snapshots; retrain both'
                )
                raise ValueError(msg)
    ids = f'{temp.artifact_id}:{precip.artifact_id}'
    return temp, precip, hashlib.sha256(ids.encode()).hexdigest()


def model_grid(
    source: Source,
    root: zarr.Group,
    corrector: TemperatureCorrector | PrecipCorrector,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Grid the models were trained on: the climatology grid, else the model source grid."""
    climate = corrector.climatology
    if climate is not None:
        return climate.lats, climate.lons
    return source.model.grid(root)


type BlockReader = Callable[[Sequence[str], NDArray[np.intp]], xr.Dataset]


@contextmanager
def open_regridded(
    source: Source,
    root: zarr.Group,
    lats: NDArray[np.float32],
    lons: NDArray[np.float32],
) -> Generator[BlockReader]:
    """Read time blocks of ``source`` on the model grid.

    CFS2 forecasts use the T126 Gaussian grid while the reanalysis models use the
    regular 0.5 degree grid, so differing grids are linearly interpolated. A margin
    around the crop keeps edge cells of the model grid inside the source grid.
    """
    with open_dataset(root.store, source.path(root)) as raw:
        view = source._crop(raw, margin=_REGRID_MARGIN)  # noqa: SLF001
        same_grid = np.array_equal(view.latitude, lats) and np.array_equal(view.longitude, lons)
        if not same_grid:
            _LOG.info(
                '%s: interpolating %dx%d source grid onto %dx%d model grid',
                source.name,
                view.sizes['latitude'],
                view.sizes['longitude'],
                len(lats),
                len(lons),
            )

        def read(variables: Sequence[str], indices: NDArray[np.intp]) -> xr.Dataset:
            block = view[list(variables)].isel(time=indices).load()
            if not same_grid:
                block = block.interp(latitude=lats, longitude=lons, method='linear')
            return block.astype(np.float32)

        yield read


def correct_forecast(
    root: zarr.Group,
    artifacts: FileStore,
    dem_store: Store,
    *,
    source: Source = CFS2_FORECAST,
) -> int:
    """Rebuild the corrected forecast with the reanalysis models; return corrected days.

    The forecast is replaced on every download, so its correction is rebuilt as a
    whole. Readers use it only when its ``forecast_updated`` matches the root
    ``updated`` marker, which is published after all days are written.
    """
    updated = root.attrs.get(const.CFS2_KEY_UPDATED)
    path = source.corrected_path(root)
    if updated is None or source.group_path not in root:
        _LOG.info('%s: no published forecast to correct', source.name)
        return 0
    temp, precip, artifact_id = load_correctors(source, artifacts, dem_store)
    lats, lons = model_grid(source, root, temp)
    if group_exists(root.store, path):
        attrs = group_attrs(root.store, path)
        if (
            attrs.get(const.CFS2_KEY_FORECAST_UPDATED) == updated
            and attrs.get('correction_artifact_id') == artifact_id
        ):
            _LOG.info('%s: correction of the %s forecast is current', source.name, updated)
            return 0
    with open_dataset(root.store, source.path(root)) as raw:
        crs = raw.rio.crs
        days = raw.sizes['time']
    if crs is None:
        msg = f'{source.name} prepared data have no CRS'
        raise ValueError(msg)
    delete_group(root.store, path)

    def blocks(read: BlockReader) -> Iterator[xr.Dataset]:
        for start in range(0, days, _FORECAST_BLOCK):
            block = read(
                (source.temp_var, source.precip_var),
                np.arange(start, min(start + _FORECAST_BLOCK, days)),
            )
            tmp = block[source.temp_var].to_numpy()
            prate = block[source.precip_var].to_numpy()
            tmp_corr = np.empty_like(tmp)
            prate_corr = np.empty_like(prate)
            for index, value in enumerate(block.time.to_numpy().astype('datetime64[D]')):
                day = date.fromisoformat(str(value))
                tmp_corr[index] = temp.correct_grid(day, tmp[index], lats, lons)['corrected']
                prate_corr[index] = precip.correct_grid(day, prate[index], lats, lons)['corrected']
            ds = _corrected_dataset(block.time, lats, lons, tmp_corr, prate_corr)
            prepared = prepare_dataset(ds, crs=crs)
            prepared.attrs['correction_artifact_id'] = artifact_id
            yield prepared

    with open_regridded(source, root, lats, lons) as read:
        write_batches(root.store, path, blocks(read))
    set_group_attrs(root.store, path, **{const.CFS2_KEY_FORECAST_UPDATED: str(updated)})
    _LOG.info('%s: corrected %d forecast days issued %s', source.name, days, updated)
    return days


def _corrected_dataset(
    time: xr.DataArray,
    lats: NDArray[np.float32],
    lons: NDArray[np.float32],
    tmp: NDArray[np.float32],
    prate: NDArray[np.float32],
) -> xr.Dataset:
    return xr.Dataset(
        {
            'TMP_corr': (
                ('time', 'latitude', 'longitude'),
                tmp,
                {'units': 'degC', 'long_name': 'Bias-corrected daily temperature'},
            ),
            'PRATE_corr': (
                ('time', 'latitude', 'longitude'),
                prate,
                {'units': 'mm/day', 'long_name': 'Bias-corrected daily precipitation'},
            ),
        },
        coords={'time': time, 'latitude': lats, 'longitude': lons},
    )


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
