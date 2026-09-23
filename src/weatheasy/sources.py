"""Source-file readers for preparation jobs, separate from prepared-Zarr access.

GRIB retains ecCodes units (e.g. K and Pa), unlike GDAL's optional metric unit
normalization. Renaming a field never converts it. Source-specific preparation
must perform and label required conversions explicitly before publication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from rasterio.crs import CRS

from weatheasy.storage import StorageError, prepare_dataset


if TYPE_CHECKING:
    from pathlib import Path


def read_grib(
    path: str | Path,
    *,
    variable: str | None = None,
    name: str | None = None,
    filter_by_keys: dict | None = None,
) -> xr.Dataset:
    """Read one field/valid time without flattening Gaussian latitude.

    Install weatheasy[prepare]. Select heterogeneous files with filter_by_keys
    (shortName, typeOfLevel, level, stepType as needed), never a numeric band ID.
    No .idx sidecar is written. Multiple valid times/levels require explicit
    selection/aggregation in a source loader, not silent squeezing here.
    """
    # The optional decoder must never be imported by prepared-data readers.
    try:
        from cfgrib.xarray_plugin import CfGribBackend
    except ImportError as err:
        msg = 'GRIB preparation requires weatheasy[prepare]'
        raise ImportError(msg) from err

    backend = {
        'indexpath': '',
        'errors': 'raise',
        'filter_by_keys': filter_by_keys or {},
        'read_keys': ['shapeOfTheEarth', 'radius', 'earthMajorAxis', 'earthMinorAxis'],
    }
    with xr.open_dataset(path, engine=CfGribBackend, backend_kwargs=backend) as raw:
        if variable is None:
            if len(raw.data_vars) != 1:
                msg = 'Select one GRIB variable explicitly'
                raise StorageError(msg)
            variable = str(next(iter(raw.data_vars)))
        field = raw[variable]
        if field.dims != ('latitude', 'longitude') or raw.valid_time.ndim != 0:
            msg = 'Select one GRIB valid time and level before preparation'
            raise StorageError(msg)
        radius = field.attrs.get('GRIB_radius')
        major = field.attrs.get('GRIB_earthMajorAxis')
        minor = field.attrs.get('GRIB_earthMinorAxis')
        if radius:
            crs = CRS.from_string(f'+proj=longlat +R={radius} +no_defs')
        elif major and minor:
            crs = CRS.from_string(f'+proj=longlat +a={major} +b={minor} +no_defs')
        else:
            msg = 'GRIB does not expose its Earth radius or ellipsoid'
            raise StorageError(msg)
        field = field.reset_coords(drop=True).expand_dims(time=[raw.valid_time.to_numpy()]).load()
        return prepare_dataset(field.to_dataset(name=name or variable), crs=crs)


def read_netcdf(
    path: str | Path,
    *,
    variables: list[str],
    crs: str | CRS,
    latitude: str = 'lat',
    longitude: str = 'lon',
) -> xr.Dataset:
    """Read actual NetCDF coordinate axes and decoded values/units, without flipping rows."""
    with xr.open_dataset(path, engine='netcdf4') as raw:
        ds = raw[variables].rename({latitude: 'latitude', longitude: 'longitude'}).load()
        return prepare_dataset(ds, crs=crs)


# NCEP GRIB2 discipline/category/number and fixed surfaces, independently of
# message ordering. Verified against CDAS pgrbh00 and CFS FLX/PGB inventories.
_CFS_FIELDS = {
    'PEVPR': (0, 1, 200, 'surface', 0, 'instant'),
    'DLWRF': (0, 5, 192, 'surface', 0, None),
    'DSWRF': (0, 4, 192, 'surface', 0, None),
    'GFLUX': (2, 0, 193, 'surface', 0, 'instant'),
    'LHTFL': (0, 0, 10, 'surface', 0, None),
    'PRATE': (0, 1, 7, 'surface', 0, None),
    'PRES': (0, 3, 0, 'surface', 0, 'instant'),
    'QMAX': (0, 1, 219, 'heightAboveGround', 2, None),
    'QMIN': (0, 1, 220, 'heightAboveGround', 2, None),
    'SHTFL': (0, 0, 11, 'surface', 0, None),
    'SNOD': (0, 1, 11, 'surface', 0, 'instant'),
    'SPFH': (0, 1, 0, 'heightAboveGround', 2, 'instant'),
    'TMAX': (0, 0, 4, 'heightAboveGround', 2, None),
    'TMIN': (0, 0, 5, 'heightAboveGround', 2, None),
    'TMP': (0, 0, 0, 'heightAboveGround', 2, 'instant'),
    'TMP_0m': (0, 0, 0, 'surface', 0, 'instant'),
    'UGRD': (0, 2, 2, 'heightAboveGround', 10, 'instant'),
    'VGRD': (0, 2, 3, 'heightAboveGround', 10, 'instant'),
    'ULWRF': (0, 5, 193, 'surface', 0, None),
    'USWRF': (0, 4, 193, 'surface', 0, None),
    'WEASD': (0, 1, 13, 'surface', 0, 'instant'),
    'RH': (0, 1, 1, 'heightAboveGround', 2, 'instant'),
}
_SOIL_LAYERS = {
    'SOILW_0-0.1m': (2, 0, 192, 0, 10),
    'SOILW_0.1-0.4m': (2, 0, 192, 10, 40),
    'SOILW_0.4-1m': (2, 0, 192, 40, 100),
    'SOILW_1-2m': (2, 0, 192, 100, 200),
    'TMP_0_0.1m': (0, 0, 0, 0, 10),
    'TMP_0.1_0.4m': (0, 0, 0, 10, 40),
    'TMP_0.4_1m': (0, 0, 0, 40, 100),
    'TMP_1-2m': (0, 0, 0, 100, 200),
}


def read_cfs2(path: str | Path, *, kind: str) -> xr.Dataset:
    """Decode a current CFS2 source, preserving provider extrema and original grids.

    TMIN/TMAX are the distinct supplied minimum/maximum-temperature parameters,
    not extrema of a shared field. In CDAS pgrbh00 their encoded interval is 0..0
    with missing statistical-processing metadata; no six-hour interval is inferred.
    Daily reduction is performed by the loader using the existing statistics.
    """
    from weatheasy import const
    from weatheasy.storage import merge_fields

    bands = const.CFS2_FLX_BANDS if kind == 'flx' else const.CFS2_PGB_BANDS
    fields = []
    for name in bands:
        if name in _SOIL_LAYERS:
            discipline, category, parameter, top, bottom = _SOIL_LAYERS[name]
            filters = {
                'discipline': discipline,
                'parameterCategory': category,
                'parameterNumber': parameter,
                'typeOfLevel': 'depthBelowLandLayer',
                'scaleFactorOfFirstFixedSurface': 2,
                'scaledValueOfFirstFixedSurface': top,
                'scaleFactorOfSecondFixedSurface': 2,
                'scaledValueOfSecondFixedSurface': bottom,
            }
        else:
            discipline, category, parameter, level_type, level, step = _CFS_FIELDS[name]
            filters = {
                'discipline': discipline,
                'parameterCategory': category,
                'parameterNumber': parameter,
                'typeOfLevel': level_type,
                'level': level,
            }
            if step:
                filters['stepType'] = step
        field = read_grib(path, name=name, filter_by_keys=filters)
        units = field[name].attrs['units']
        values = field[name].astype('float32')
        if name.startswith('TMP') or name in ('TMIN', 'TMAX'):
            if units != 'K':
                msg = f'Expected K for {name}, got {units}'
                raise StorageError(msg)
            values = values - 273.15
            units = 'degC'
        # Preserve the existing numeric pressure contract (GDAL returns Pa),
        # without silently adopting the old catalogue's incorrect hPa label.
        if name == 'PEVPR':
            # NCEP local parameter 0/1/200 is W/m² even when ecCodes reports unknown.
            # Four instantaneous samples yield an approximate daily evaporation.
            values = values * (86400 / 2.45e6)
            units = 'mm/day'
        values.attrs = {'units': units}
        if name == 'PEVPR':
            values.attrs['comment'] = (
                'Approximate daily evaporation from the mean of instantaneous '
                '00/06/12/18 UTC samples; not an exact daily accumulation.'
            )
        field[name] = values
        fields.append(field)
    return merge_fields(*fields)
