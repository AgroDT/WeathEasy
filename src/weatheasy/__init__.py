from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from rasterio.coords import BoundingBox

from weatheasy import const
from weatheasy.error import CFS2Error, CMIP6DateRangeError, CoordsError, DateRangeError
from weatheasy.storage import StorageError, open_dataset, select_point


try:
    from weatheasy._version import version

    __version__: str | None = version
except ImportError:
    __version__ = None


if TYPE_CHECKING:
    from collections.abc import Sequence

    import zarr
    from numpy.typing import NDArray


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
    """Read prepared daily fields, using each field's actual grid coordinates."""
    _check_date_range(begin, end)
    try:
        updated = root[const.CFS2_DIR].attrs[const.CFS2_KEY_UPDATED]
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
            path = f'cfs2/{kind}/{grid}'
            if kind == 'reanalysis':
                target = cast('zarr.Group', root[path])
                # The loader creates empty groups before trying source files.
                # Only pristine groups mean no data; incomplete/corrupt stores
                # must still fail validation in open_dataset.
                if not target.attrs and not any(target.keys()):
                    continue
            values = _read_point(root, path, first, last, coords, selected)
            offset = (first - begin).days
            for index, variable in enumerate(selected):
                result[variables.index(variable), offset : offset + values.shape[1]] = values[index]
    return result


def get_cmip6_data(
    *,
    root: zarr.Group,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    _check_date_range(begin, end)
    first = date(const.CMIP6_FIRST_YEAR, 1, 1)
    last = date(const.CMIP6_LAST_YEAR, 12, 31)
    for value in (begin, end):
        if not first <= value <= last:
            raise CMIP6DateRangeError(value, first, last)
    return np.stack(
        [
            _read_point(root, f'cmip6/{variable}', begin, end, coords, [variable])[0]
            for variable in variables
        ]
    )


def get_era5_data(
    *,
    root: zarr.Group,
    begin: date,
    end: date,
    coords: Coords,
    variables: Sequence[str],
) -> NDArray[np.float32]:
    """Read completed ERA5 months, including the current incomplete year."""
    from weatheasy.era5 import available_period
    from weatheasy.error import ERA5DateRangeError

    _check_date_range(begin, end)
    first, last = available_period(root)
    for value in (begin, end):
        if not first <= value <= last:
            raise ERA5DateRangeError(value, first, last)
    return _read_point(root, 'era5/daily', begin, end, coords, variables)


def _read_point(
    root: zarr.Group, group: str, begin: date, end: date, coords: Coords, variables: Sequence[str]
):
    dates = np.arange(np.datetime64(begin), np.datetime64(end) + np.timedelta64(1, 'D'))
    with open_dataset(root.store, '/'.join(filter(None, (root.path, group)))) as dataset:
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
