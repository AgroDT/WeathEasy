"""Field-at-a-time daily reduction of ERA5-Land hourly and accumulated inputs.

Accumulated fields use the following midnight, including at month/year boundaries.
PEVPR is open-water evaporation, not FAO reference ET0. Evaporation and turbulent
heat fluxes are positive upwards; source signs are reversed without clipping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from weatheasy import const
from weatheasy.storage import StorageError, prepare_dataset


if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date


UNITS = {
    'TMP': 'degC',
    'TMIN': 'degC',
    'TMAX': 'degC',
    'UGRD': 'm/s',
    'VGRD': 'm/s',
    'PRES': 'hPa',
    'PRATE': 'kg m-2 s-1',
    'SNOD': 'm',
    'WEASD': 'kg m-2',
    'LHTFL': 'W m-2',
    'SHTFL': 'W m-2',
    'DSWRF': 'W m-2',
    'DLWRF': 'W m-2',
    'PEVPR': 'mm/day',
    'WIND': 'm/s',
    'RH': '%',
    'SPFH': 'kg/kg',
    'QMIN': 'kg/kg',
    'QMAX': 'kg/kg',
    'USWRF': 'W m-2',
    'ULWRF': 'W m-2',
}
UNITS.update(
    {
        name: 'degC' if name.startswith('TMP') else 'm3 m-3'
        for name in const.ERA5_VARS
        if name.startswith(('TMP_', 'SOILW_'))
    }
)


def process_day(main: xr.Dataset, boundary: xr.Dataset, day: date) -> xr.Dataset:
    """Reduce one spatial field at a time; memory is independent of month length.

    A full field is a natural GRIB decoding block, spanning multiple Zarr chunks
    (with partial chunks only at grid edges). Daily reductions are collected
    into bounded time blocks before writing. Shared hourly inputs are decoded
    once for all derived variables.
    """
    start = np.datetime64(day, 'h')
    following = start + np.timedelta64(24, 'h')

    last = main.time.to_numpy()[-1]

    def read(name: str, stamp: np.datetime64):
        source = main if stamp <= last else boundary
        if stamp not in source.time:
            return np.full((source.sizes['latitude'], source.sizes['longitude']), np.nan)
        return source[name].sel(time=stamp).to_numpy().astype(np.float64, copy=False)

    def midnight(name: str):
        return read(name, following)

    values = {}
    shared = {'t2m', 'd2m', 'sp', 'u10', 'v10'}
    for name, spec in const.ERA5_VARS.items():
        if spec.nc_name in shared:
            continue
        if spec.accumulated:
            reduced = midnight(spec.nc_name)
        else:
            reduced = _reduce_grib_field(read, spec, start)
        values[name] = (reduced * spec.scale + spec.offset).astype(np.float32)
        del reduced
    values.update(_shared_daily_values(read, start))
    values['USWRF'] = ((midnight('ssrd') - midnight('ssr')) / 86400).astype(np.float32)
    values['ULWRF'] = ((midnight('strd') - midnight('str')) / 86400).astype(np.float32)
    result = _prepare_daily_values(values, day, main.latitude, main.longitude)
    source_grib = {
        label: {
            name: {
                'edition': field.attrs.get('GRIB_edition'),
                'radius': field.attrs.get('GRIB_radius'),
            }
            for name, field in source.data_vars.items()
            if 'GRIB_radius' in field.attrs
        }
        for label, source in (('main', main), ('boundary', boundary))
    }
    if any(source_grib.values()):
        result.attrs['source_grib'] = source_grib
    return result


def _reduce_grib_field(read, spec: const.Era5Var, start: np.datetime64):  # noqa: ANN001
    reducers: dict[Callable | None, np.ufunc] = {
        np.mean: np.add,
        np.max: np.maximum,
        np.min: np.minimum,
    }
    combine = reducers[spec.daily_stat]
    reduced = read(spec.nc_name, start).copy()
    for hour in range(1, 24):
        field = read(spec.nc_name, start + np.timedelta64(hour, 'h'))
        combine(reduced, field, out=reduced)
        del field
    return reduced / 24 if spec.daily_stat is np.mean else reduced


def _shared_daily_values(read, start: np.datetime64) -> dict:  # noqa: ANN001
    totals = {}
    for hour in range(24):
        stamp = start + np.timedelta64(hour, 'h')
        t = read('t2m', stamp) - 273.15
        td = read('d2m', stamp) - 273.15
        pressure = read('sp', stamp)
        u = read('u10', stamp)
        v = read('v10', stamp)
        es = 611.21 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))
        ea = 611.21 * np.exp((18.678 - td / 234.5) * td / (257.14 + td))
        q = 0.622 * ea / np.maximum(pressure - 0.378 * ea, 1.0)
        hourly = {
            'TMP': t,
            'TMIN': t,
            'TMAX': t,
            'PRES': pressure * 0.01,
            'UGRD': u,
            'VGRD': v,
            'WIND': np.hypot(u, v),
            'RH': np.clip(100 * ea / np.maximum(es, 1e-8), 0, 100),
            'SPFH': q,
            'QMIN': q,
            'QMAX': q,
        }
        for name, field in hourly.items():
            if not hour:
                totals[name] = field.copy()
            elif name in ('TMIN', 'QMIN'):
                np.minimum(totals[name], field, out=totals[name])
            elif name in ('TMAX', 'QMAX'):
                np.maximum(totals[name], field, out=totals[name])
            else:
                totals[name] += field
        del hourly, field, t, td, pressure, u, v, es, ea, q
    return {
        name: (field if name in {'TMIN', 'TMAX', 'QMIN', 'QMAX'} else field / 24).astype(np.float32)
        for name, field in totals.items()
    }


def validate_boundary(main: xr.Dataset, boundary: xr.Dataset) -> None:
    if any(not np.array_equal(main[axis], boundary[axis]) for axis in ('latitude', 'longitude')):
        msg = 'ERA5 month and following midnight have different coordinates'
        raise StorageError(msg)


def _prepare_daily_values(values: dict, day: date, latitudes, longitudes):  # noqa: ANN001
    fields = {}
    for name, value in values.items():
        info = (
            const.ERA5_VARS[name].info if name in const.ERA5_VARS else const.ERA5_DERIVED_INFO[name]
        )
        attrs = {'units': UNITS[name], 'long_name': info.en}
        attrs['cell_methods'] = 'time: mean'
        if name in const.ERA5_VARS:
            spec = const.ERA5_VARS[name]
            attrs['cell_methods'] = (
                'time: sum'
                if name == 'PEVPR'
                else f'time: {spec.daily_stat.__name__}'
                if spec.daily_stat
                else 'time: mean'
            )
        fields[name] = (('time', 'latitude', 'longitude'), value[np.newaxis], attrs)
    ds = xr.Dataset(
        fields,
        coords={
            'time': [np.datetime64(day)],
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )
    return prepare_dataset(ds, crs=const.ERA5_CRS)
