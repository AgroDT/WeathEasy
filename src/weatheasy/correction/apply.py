"""Apply trained models to committed daily reanalysis as an external job."""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .precip import PrecipCorrector
from .temperature import TemperatureCorrector
from weatheasy.storage import last_date, open_dataset, prepare_dataset, write_batch


if TYPE_CHECKING:
    import zarr
    from zarr.abc.store import Store

    from .artifacts import Artifacts
    from .sources import Source


_LOG = logging.getLogger(__name__)
_FIELDS = {'TMP_corr', 'PRATE_corr'}


def correct_store(  # noqa: C901, PLR0912, PLR0915
    source: Source,
    root: zarr.Group,
    artifacts: Artifacts,
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
        del root[source.correction_path]
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
    for day in pending:
        temperature = temp.correct(root, day)['corrected']
        precipitation = precip.correct(root, day)['corrected']
        if temperature.shape != (len(lats), len(lons)) or precipitation.shape != temperature.shape:
            msg = f'{source.name}: corrected grid shape differs from source on {day}'
            raise ValueError(msg)
        ds = xr.Dataset(
            {
                'TMP_corr': (
                    ('time', 'latitude', 'longitude'),
                    temperature[None].astype(np.float32),
                    {'units': 'degC', 'long_name': 'Bias-corrected daily temperature'},
                ),
                'PRATE_corr': (
                    ('time', 'latitude', 'longitude'),
                    precipitation[None].astype(np.float32),
                    {'units': 'mm/day', 'long_name': 'Bias-corrected daily precipitation'},
                ),
            },
            coords={'time': [np.datetime64(day)], 'latitude': lats, 'longitude': lons},
            attrs={'correction_artifact_id': artifact_id},
        )
        prepared = prepare_dataset(ds, crs=crs)
        prepared.attrs['correction_artifact_id'] = artifact_id
        write_batch(root.store, path, prepared)
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
