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


def read_grib(path: Path, expected: np.ndarray, required: set[str]) -> xr.Dataset:
    from cfgrib.xarray_plugin import CfGribBackend  # noqa: PLC0415

    if path.stat().st_size == 0:
        msg = f'Missing ERA5 variables: {sorted(required)}'
        raise StorageError(msg)
    fields = []
    # Separate variables have different heights/soil levels. Share one header
    # index across their backends, without retaining .idx files in the raw cache.
    with TemporaryDirectory(prefix='weatheasy-grib-') as index_dir:
        indexpath = f'{index_dir}/{{short_hash}}.idx'
        _validate_headers(path, indexpath, required)
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
                if not np.array_equal(field.valid_time.to_numpy(), expected):
                    msg = f'Incomplete or unexpected ERA5 timestamps for {name}'
                    raise StorageError(msg)
                fields.append(field.reset_coords(drop=True).rename(valid_time='time'))
    return xr.merge(fields, join='exact', compat='identical').rio.write_crs(_field_crs(fields[0]))


def _validate_headers(path: Path, indexpath: str, required: set[str]) -> None:
    from cfgrib import FileStream, compute_index_keys  # noqa: PLC0415
    from cfgrib.dataset import open_fileindex  # noqa: PLC0415

    keys = compute_index_keys(('valid_time',), filter_by_keys={'cfVarName': ''})
    index = open_fileindex(FileStream(str(path), errors='raise'), indexpath, keys)
    names = set(index['cfVarName'])
    if missing := required - names:
        msg = f'Missing ERA5 variables: {sorted(missing)}'
        raise StorageError(msg)
    if unexpected := names - required:
        msg = f'Unexpected ERA5 GRIB variables: {sorted(unexpected)}'
        raise StorageError(msg)
    if index['gridType'] != ['regular_ll'] or len(index['md5GridSection']) != 1:
        msg = 'ERA5 GRIB requires one regular latitude/longitude grid and CRS'
        raise StorageError(msg)
    seen = set()
    for headers, messages in index.iter_index():
        key = (headers[keys.index('cfVarName')], headers[keys.index('valid_time')])
        if key in seen or len(messages) != 1:
            msg = f'Duplicate ERA5 GRIB timestamp for {key[0]}'
            raise StorageError(msg)
        seen.add(key)


def _field_crs(field: xr.DataArray) -> CRS:
    radius = field.attrs.get('GRIB_radius')
    if radius:
        return CRS.from_string(f'+proj=longlat +R={radius} +no_defs')
    major = field.attrs['GRIB_earthMajorAxis']
    minor = field.attrs['GRIB_earthMinorAxis']
    return CRS.from_string(f'+proj=longlat +a={major} +b={minor} +no_defs')
