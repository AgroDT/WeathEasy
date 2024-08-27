from __future__ import annotations

from datetime import date
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from weatheasy import const
from weatheasy.error import CFS2Error, CMIP6DateRangeError, CoordsError, DateRangeError
from weatheasy.util import utc_now


if TYPE_CHECKING:
    from collections.abc import Sequence

    import zarr
    from numpy.typing import NDArray
    from rasterio.coords import BoundingBox


class Coords(NamedTuple):
    latitude: float
    longitude: float


def get_cfs2_data(
    *,
    root: zarr.Group,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    _check_date_range(begin, end)
    today = utc_now().date()

    cfs2_group = root.require_group(const.CFS2_DIR)
    try:
        cfs2_updated = cfs2_group.attrs[const.CFS2_KEY_UPDATED]
    except KeyError as err:
        raise CFS2Error from err
    first_forecast_day = date.fromisoformat(cfs2_updated) - const.CFS2_REANALYSIS_LAST_DATE_OFFSET

    forecast_group = root.require_group(const.CFS2_FORECAST_DIR)
    if begin >= today:
        return _get_cfs2_forecast(forecast_group, first_forecast_day, begin, end, coords, variables)

    reanalysis_group = root.require_group(const.CFS2_REANALYSIS_DIR)
    if end <= today:
        return _get_cfs2_reanalysis(
            reanalysis_group, const.CFS2_REANALYSIS_FIRST_DATE, begin, end, coords, variables
        )

    mid = min(today, end)
    reanalysis = _get_cfs2_reanalysis(
        reanalysis_group, const.CFS2_REANALYSIS_FIRST_DATE, begin, mid, coords, variables
    )
    forecast = _get_cfs2_forecast(
        forecast_group, first_forecast_day, mid + const.ONE_DAY, end, coords, variables
    )

    return np.concat((reanalysis, forecast), axis=1)


def get_cmip6_data(
    *,
    root: zarr.Group,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    _check_date_range(begin, end)
    first_date = date(const.CMIP6_FIRST_YEAR, 1, 1)
    last_date = date(const.CMIP6_LAST_YEAR, 12, 31)
    if begin < first_date:
        raise CMIP6DateRangeError(begin, first_date, last_date)
    if end > last_date:
        raise CMIP6DateRangeError(end, first_date, last_date)

    begin_i = (begin - first_date).days
    end_i = (end - first_date).days + 1

    lat_i, lon_i = _coords_to_indices(coords, const.CMIP6_RESOLUTION, const.CMIP6_BBOX, lon360=True)

    cmip6 = root.require_group('cmip6')
    res = np.zeros((len(variables), end_i - begin_i), np.float32)
    for var_i, var in enumerate(variables):
        res[var_i] = cmip6[var][begin_i:end_i, lat_i, lon_i]
    return res


def _check_date_range(first: date, last: date):
    if first > last:
        raise DateRangeError


def _coords_to_indices(
    coords: Coords,
    resolution: float,
    bbox: BoundingBox,
    *,
    lon360: bool = False,
) -> tuple[int, int]:
    if lon360 and coords.longitude < 0:
        coords = Coords(coords.latitude, coords.longitude + 360)
    if not (
        bbox.left <= coords.longitude <= bbox.right
        and bbox.bottom <= coords.latitude <= bbox.top
    ):  # fmt: skip
        raise CoordsError(coords, bbox)

    return (
        round((bbox.top - coords.latitude) / resolution),
        round((coords.longitude - bbox.left) / resolution),
    )


def _get_cfs2_data(
    group: zarr.Group,
    first_date: date,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
    *,
    flx_bbox: BoundingBox,
    flx_resolution: float,
    pgb_bbox: BoundingBox,
    pbg_resolution: float,
):
    pgb_coord_ind = _coords_to_indices(coords, pbg_resolution, pgb_bbox)
    flx_coord_ind = _coords_to_indices(coords, flx_resolution, flx_bbox, lon360=True)

    end += const.ONE_DAY
    begin_i = (begin - first_date).days
    end_i = (end - first_date).days

    res = np.zeros((len(variables), (end - begin).days), np.float32)
    for var_i, var in enumerate(variables):
        coord_ind = pgb_coord_ind if var in const.CFS2_PGB_BANDS else flx_coord_ind
        chunk = group[var][begin_i:end_i, *coord_ind]
        res[var_i] = chunk if chunk.size else np.nan

    return res


_get_cfs2_reanalysis = partial(
    _get_cfs2_data,
    flx_bbox=const.CFS2_REANALYSIS_BBOX,
    flx_resolution=const.CFS2_REANALYSIS_RESOLUTION[0],
    pgb_bbox=const.CFS2_REANALYSIS_BBOX,
    pbg_resolution=const.CFS2_REANALYSIS_RESOLUTION[0],
)

_get_cfs2_forecast = partial(
    _get_cfs2_data,
    flx_bbox=const.CFS2_FLX_BBOX,
    flx_resolution=const.CFS2_FLX_RESOLUTION[0],
    pgb_bbox=const.CFS2_PGB_BBOX,
    pbg_resolution=const.CFS2_PGB_RESOLUTION[0],
)
