"""Source-file readers for preparation jobs, separate from prepared-Zarr access.

CRS and units come from the source product's contract. File metadata is retained
for diagnostics; coordinates, timestamps and decoded values come from the file.
Renaming a field never converts it. Preparation performs conversions explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from weatheasy import const
from weatheasy.storage import StorageError, merge_fields, prepare_dataset


if TYPE_CHECKING:
    from pathlib import Path

    from rasterio import CRS


def read_grib(
    path: str | Path,
    *,
    crs: str | CRS,
    units: str,
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
        from cfgrib.xarray_plugin import CfGribBackend  # noqa: PLC0415
    except ImportError as err:
        msg = 'GRIB preparation requires weatheasy[prepare]'
        raise ImportError(msg) from err

    backend = {
        'indexpath': '',
        'errors': 'raise',
        'filter_by_keys': filter_by_keys or {},
        'read_keys': ['edition', 'radius'],
    }
    with xr.open_dataset(path, engine=CfGribBackend, backend_kwargs=backend) as raw:
        if variable is None:
            if len(raw.data_vars) != 1:
                msg = 'Select one GRIB variable explicitly'
                raise StorageError(msg)
            variable = str(next(iter(raw.data_vars)))
        field = raw[variable]
        source_grib = {
            'edition': field.attrs.get('GRIB_edition'),
            'radius': field.attrs.get('GRIB_radius'),
            'units': field.attrs.get('units'),
        }
        field = field.reset_coords(drop=True).expand_dims(time=[raw.valid_time.to_numpy()]).load()
        field.attrs = {'units': units}
        result = prepare_dataset(field.to_dataset(name=name or variable), crs=crs)
        result.attrs['source_grib'] = {name or variable: source_grib}
        return result


def read_netcdf(
    path: str | Path,
    *,
    units: dict[str, str],
    crs: str | CRS,
    latitude: str = 'lat',
    longitude: str = 'lon',
) -> xr.Dataset:
    """Read actual axes/values and label them using the source product's contract."""
    with xr.open_dataset(path, engine='netcdf4') as raw:
        ds = raw[list(units)].rename({latitude: 'latitude', longitude: 'longitude'}).load()
        source_units = {name: ds[name].attrs.get('units') for name in units}
        for name, unit in units.items():
            ds[name].attrs = {'units': unit}
        result = prepare_dataset(ds, crs=crs)
        result.attrs['source_units'] = source_units
        return result


# NCEP GRIB2 discipline/category/number and fixed surfaces, independently of
# message ordering. Verified against CDAS pgrbh00 and CFS FLX/PGB inventories.
# Fluxes/rates use interval averages; CDAS may also contain instantaneous copies.
_CFS_FIELDS = {
    'PEVPR': (0, 1, 200, 'surface', 0, 'instant'),
    'DLWRF': (0, 5, 192, 'surface', 0, 'avg'),
    'DSWRF': (0, 4, 192, 'surface', 0, 'avg'),
    'GFLUX': (2, 0, 193, 'surface', 0, 'instant'),
    'LHTFL': (0, 0, 10, 'surface', 0, 'avg'),
    'PRATE': (0, 1, 7, 'surface', 0, 'avg'),
    'PRES': (0, 3, 0, 'surface', 0, 'instant'),
    'QMAX': (0, 1, 219, 'heightAboveGround', 2, None),
    'QMIN': (0, 1, 220, 'heightAboveGround', 2, None),
    'SHTFL': (0, 0, 11, 'surface', 0, 'avg'),
    'SNOD': (0, 1, 11, 'surface', 0, 'instant'),
    'SPFH': (0, 1, 0, 'heightAboveGround', 2, 'instant'),
    'TMAX': (0, 0, 4, 'heightAboveGround', 2, None),
    'TMIN': (0, 0, 5, 'heightAboveGround', 2, None),
    'TMP': (0, 0, 0, 'heightAboveGround', 2, 'instant'),
    'TMP_0m': (0, 0, 0, 'surface', 0, 'instant'),
    'UGRD': (0, 2, 2, 'heightAboveGround', 10, 'instant'),
    'VGRD': (0, 2, 3, 'heightAboveGround', 10, 'instant'),
    'ULWRF': (0, 5, 193, 'surface', 0, 'avg'),
    'USWRF': (0, 4, 193, 'surface', 0, 'avg'),
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
    bands = const.CFS2_FLX_BANDS if kind == 'flx' else const.CFS2_PGB_BANDS
    fields = []
    source_grib = {}
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
        field = read_grib(
            path,
            name=name,
            filter_by_keys=filters,
            crs=const.CFS2_CRS,
            units=const.CFS2_SOURCE_UNITS[name],
        )
        source_grib.update(field.attrs.pop('source_grib'))
        units = field[name].attrs['units']
        values = field[name].astype('float32')
        if name.startswith('TMP') or name in ('TMIN', 'TMAX'):
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
    result = merge_fields(*fields)
    result.attrs['source_grib'] = source_grib
    return result
