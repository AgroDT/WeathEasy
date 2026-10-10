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
                    'read_keys': ['edition', 'radius'],
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
    # ECMWF uses R=6371229 in the model/GRIB2, but GRIB1 encodes R=6367470.
    # These describe the same ERA5-Land nodes; normalize metadata, not values.
    # https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
    radius = field.attrs.get('GRIB_radius')
    if field.attrs.get('GRIB_gridType') != 'regular_ll' or radius not in (6367470, 6371229):
        msg = f'Unsupported ERA5 GRIB grid and CRS: {field.attrs.get("GRIB_gridType")}, R={radius}'
        raise StorageError(msg)
    return CRS.from_string('+proj=longlat +R=6371229 +no_defs')
