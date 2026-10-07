"""Read prepared weather datasets at a requested point and date range."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rasterio.coords import BoundingBox

from weatheasy import const
from weatheasy.error import (
    CFS2Error,
    CMIP6DateRangeError,
    CoordsError,
    DateRangeError,
    ERA5DateRangeError,
)
from weatheasy.storage import (
    StorageError,
    group_attrs,
    open_dataset,
    pristine_group,
    select_point,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from zarr.abc.store import Store


class Coords(NamedTuple):
    latitude: float
    longitude: float


def get_cfs2_data(  # noqa: C901
    *,
    store: Store,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    """Read prepared daily fields; PRATE and PRATE_corr are returned in mm/day."""
    _check_date_range(begin, end)
    try:
        updated = group_attrs(store, '')[const.CFS2_KEY_UPDATED]
    except KeyError as err:
        raise CFS2Error from err
    boundary = date.fromisoformat(str(updated)) - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
    result = np.full((len(variables), (end - begin).days + 1), np.nan, np.float32)
    for kind, first, last in (
        ('reanalysis', begin, min(end, boundary - const.ONE_DAY)),
        ('forecast', max(begin, boundary), end),
    ):
        if first > last:
            continue
        for grid, bands in (('flx', const.CFS2_FLX_BANDS), ('pgb', const.CFS2_PGB_BANDS)):
            selected = [var for var in variables if var in bands]
            if not selected:
                continue
            path = f'{kind}/{grid}'
            # The loader creates empty groups before trying source files.
            # Only pristine groups mean no data; incomplete/corrupt stores
            # must still fail validation in open_dataset.
            if kind == 'reanalysis' and pristine_group(store, path):
                continue
            values = _read_point(store, path, first, last, coords, selected)
            offset = (first - begin).days
            for index, variable in enumerate(selected):
                result[variables.index(variable), offset : offset + values.shape[1]] = values[index]
        if kind == 'reanalysis':
            corrected = [var for var in variables if var in ('TMP_corr', 'PRATE_corr')]
            if corrected:
                values = _read_point(
                    store,
                    'reanalysis/correction',
                    first,
                    last,
                    coords,
                    corrected,
                )
                offset = (first - begin).days
                for index, variable in enumerate(corrected):
                    result[variables.index(variable), offset : offset + values.shape[1]] = values[
                        index
                    ]
    return _daily_precipitation(result, variables, 'PRATE', 21600)


def get_cmip6_data(
    *,
    store: Store,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    """Read daily CMIP6 fields; precipitation pr is returned in mm/day."""
    _check_date_range(begin, end)
    first = date(const.CMIP6_FIRST_YEAR, 1, 1)
    last = date(const.CMIP6_LAST_YEAR, 12, 31)
    for value in (begin, end):
        if not first <= value <= last:
            raise CMIP6DateRangeError(value, first, last)
    result = np.stack(
        [_read_point(store, variable, begin, end, coords, [variable])[0] for variable in variables]
    )
    return _daily_precipitation(result, variables, 'pr', 86400)


def get_era5_data(
    *,
    store: Store,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    """Read completed ERA5 months; PRATE and PRATE_corr are returned in mm/day."""
    # Load the preparation module only when ERA5 data is requested.
    from weatheasy.era5 import available_period  # noqa: PLC0415

    _check_date_range(begin, end)
    first, last = available_period(store)
    for value in (begin, end):
        if not first <= value <= last:
            raise ERA5DateRangeError(value, first, last)
    raw = [var for var in variables if var not in ('TMP_corr', 'PRATE_corr')]
    corrected = [var for var in variables if var in ('TMP_corr', 'PRATE_corr')]
    result = np.full((len(variables), (end - begin).days + 1), np.nan, np.float32)
    for group, selected in (('daily', raw), ('correction', corrected)):
        if selected:
            values = _read_point(store, group, begin, end, coords, selected)
            for index, variable in enumerate(selected):
                result[variables.index(variable)] = values[index]
    return _daily_precipitation(result, variables, 'PRATE', 86400)


def _daily_precipitation(
    result: NDArray[np.float32], variables: Sequence[str], name: str, factor: int
) -> NDArray[np.float32]:
    # CFS2 stores the sum of four six-hour rates; ERA5/CMIP6 store a daily rate.
    # Corrected precipitation is already mm/day and never enters this conversion.
    for index, variable in enumerate(variables):
        if variable == name:
            result[index] *= factor
    return result


def _read_point(  # noqa: PLR0917
    store: Store,
    group: str,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
):
    dates = np.arange(np.datetime64(begin), np.datetime64(end) + np.timedelta64(1, 'D'))
    with open_dataset(store, group) as dataset:
        try:
            point = select_point(dataset, latitude=coords.latitude, longitude=coords.longitude)
        except StorageError as err:
            bbox = BoundingBox(
                float(dataset.longitude.min()),
                float(dataset.latitude.min()),
                float(dataset.longitude.max()),
                float(dataset.latitude.max()),
            )
            raise CoordsError(coords, bbox) from err
        return (
            point[variables]
            .reindex(time=dates)
            .to_array()
            .transpose('variable', 'time')
            .to_numpy()
            .astype(np.float32)
        )


def _check_date_range(first: date, last: date):
    if first > last:
        raise DateRangeError
