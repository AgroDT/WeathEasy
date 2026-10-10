"""Open ERA5-Land GRIB fields lazily with the standard xarray/cfgrib backend."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from rasterio import CRS

from weatheasy.storage import StorageError


if TYPE_CHECKING:
    from pathlib import Path


def read_grib(path: Path, required: set[str]) -> xr.Dataset:
    from cfgrib.xarray_plugin import CfGribBackend  # noqa: PLC0415

    fields = []
    # Separate variables have different heights/soil levels. Share one header
    # index across their backends, without retaining .idx files in the raw cache.
    with TemporaryDirectory(prefix='weatheasy-grib-') as index_dir:
        indexpath = f'{index_dir}/{{short_hash}}.idx'
        for name in sorted(required):
            with xr.open_dataset(
                path,
                engine=CfGribBackend,
                cache=False,
                backend_kwargs={
                    'indexpath': indexpath,
                    'filter_by_keys': {'cfVarName': name},
                    'time_dims': ('valid_time',),
                    'squeeze': False,
                    'read_keys': ['radius', 'earthMajorAxis', 'earthMinorAxis'],
                    'cache_geo_coords': False,
                    'errors': 'raise',
                    'values_dtype': np.dtype('float64'),
                },
            ) as ds:
                field = ds[name].squeeze(
                    [
                        dim
                        for dim in ds[name].dims
                        if dim not in ('valid_time', 'latitude', 'longitude')
                    ],
                    drop=True,
                )
                if fields and (
                    _field_crs(field) != _field_crs(fields[0])
                    or any(
                        not np.array_equal(field[axis].to_numpy(), fields[0][axis].to_numpy())
                        for axis in ('latitude', 'longitude')
                    )
                ):
                    msg = 'ERA5 GRIB requires one regular latitude/longitude grid and CRS'
                    raise StorageError(msg)
                fields.append(field.reset_coords(drop=True).rename(valid_time='time'))
    return xr.merge(fields, join='exact', compat='identical').rio.write_crs(_field_crs(fields[0]))


def _field_crs(field: xr.DataArray) -> CRS:
    radius = field.attrs.get('GRIB_radius')
    if radius:
        return CRS.from_string(f'+proj=longlat +R={radius} +no_defs')
    major = field.attrs['GRIB_earthMajorAxis']
    minor = field.attrs['GRIB_earthMinorAxis']
    return CRS.from_string(f'+proj=longlat +a={major} +b={minor} +no_defs')
