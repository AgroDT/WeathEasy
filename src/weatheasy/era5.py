"""ERA5-Land monthly GRIB jobs with bounded, field-at-a-time daily reduction.

Accumulated fields use the following midnight, including at month/year boundaries.
PEVPR is open-water evaporation, not FAO reference ET0. Evaporation and turbulent
heat fluxes are positive upwards; source signs are reversed without clipping.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from weatheasy import const
from weatheasy.era5_grib import read_grib
from weatheasy.raw_cache import DownloadCache, download_cache
from weatheasy.storage import (
    StorageError,
    group_attrs,
    group_exists,
    last_date,
    open_dataset,
    prepare_dataset,
    set_group_attrs,
    write_batches,
)
from weatheasy.util import utc_now


if TYPE_CHECKING:
    from zarr.abc.store import Store

    from weatheasy.file_storage import FileStore


EXTRA = {
    'd2m': '2m_dewpoint_temperature',
    'ssr': 'surface_net_solar_radiation',
    'str': 'surface_net_thermal_radiation',
}
LOGGER = logging.getLogger(__name__)
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


def next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def available_period(store: Store) -> tuple[date, date]:
    """Return the committed time range, including incomplete years.

    last_month is a convenience marker. A crash after committing data but before
    updating that marker must neither hide data nor cause a duplicate append.
    """
    with open_dataset(store, 'daily') as ds:
        times = ds.time.to_numpy().astype('datetime64[D]')
        first = date.fromisoformat(str(times[0]))
        last = date.fromisoformat(str(times[-1]))
    return first, last


def download_era5_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Append complete months through the last month with available midnight data.

    bbox is (north, west, south, east), without antimeridian crossing. The CDS
    publication lag is five days; only whole months are requested. Run as an
    external job. No download is triggered by API or prepared-data reads.
    """
    bbox = bbox or (90.0, -180.0, -90.0, 180.0)
    north, west, south, east = bbox
    if not np.isfinite(bbox).all() or not (
        -90 <= south < north <= 90 and -180 <= west < east <= 180
    ):
        msg = 'Invalid ERA5 bbox (north, west, south, east)'
        raise StorageError(msg)
    if (
        group_exists(store, '')
        and (stored_bbox := group_attrs(store, '').get('request_bbox')) is not None
        and stored_bbox != list(bbox)
    ):
        msg = 'Existing ERA5 store has a different requested bbox'
        raise StorageError(msg)
    set_group_attrs(store, '', request_bbox=list(bbox))
    path = 'daily'
    previous = last_date(store, path)
    first = next_month(previous) if previous else date(const.ERA5_FIRST_YEAR, 1, 1)
    latest = utc_now().date() - 5 * const.ONE_DAY
    if previous:
        set_group_attrs(store, '', last_month=[previous.year, previous.month])
    if next_month(first) > latest:
        return
    try:
        # The CDS client is optional and needed only for a pending download.
        import cdsapi  # noqa: PLC0415
    except ImportError as err:
        msg = 'ERA5 preparation requires weatheasy[prepare]'
        raise ImportError(msg) from err
    client = cdsapi.Client()
    with download_cache(download_dir, 'era5') as cache:
        downloader = _Downloader(client, cache, bbox)
        while next_month(first) <= latest:
            following = next_month(first)
            LOGGER.info('Preparing ERA5-Land %s', first.strftime('%Y-%m'))
            main = downloader.retrieve(first, boundary=False)
            boundary = downloader.retrieve(following, boundary=True)
            _validate_boundary(main, boundary)
            _validate_bbox(main, bbox)
            write_batches(store, path, _daily_batches(main, boundary, first, bbox, cache))
            set_group_attrs(store, '', last_month=[first.year, first.month])
            first = following


class _Downloader:
    __slots__ = ('bbox', 'cache', 'client')

    def __init__(self, client: Any, cache: DownloadCache | Path, bbox: tuple) -> None:
        self.client = client
        self.cache = DownloadCache(cache) if isinstance(cache, Path) else cache
        self.bbox = bbox

    def retrieve(self, month: date, *, boundary: bool) -> xr.Dataset:
        names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
        names.update({cds: nc for nc, cds in EXTRA.items()})
        variables = sorted(names)
        pieces = []
        # Retain bounded request sizes; GRIB skips the server's NetCDF conversion.
        for offset in range(0, len(variables), 8):
            subset = variables[offset : offset + 8]
            pieces.append(
                self._retrieve_part(
                    month,
                    subset,
                    {names[name] for name in subset},
                    boundary=boundary,
                )
            )
        return xr.merge(pieces, join='exact', compat='identical')

    def _retrieve_part(
        self,
        month: date,
        variables: list[str],
        required: set[str],
        *,
        boundary: bool,
    ) -> xr.Dataset:
        if boundary:
            day = ['01']
            time_ = ['00:00']
        else:
            day_count = calendar.monthrange(month.year, month.month)[1]
            day = [f'{d:02d}' for d in range(1, day_count + 1)]
            time_ = [f'{h:02d}:00' for h in range(24)]
        request = {
            'variable': variables,
            'year': str(month.year),
            'month': f'{month.month:02d}',
            'day': day,
            'time': time_,
            'area': list(self.bbox),
            'data_format': 'grib',
            'download_format': 'unarchived',
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache.file(f'{month:%Y%m}-{key}.grib')
        start = np.datetime64(month, 'h')
        expected = (
            np.array([start])
            if boundary
            else np.arange(start, np.datetime64(next_month(month), 'h'), np.timedelta64(1, 'h'))
        )
        if not path.exists():
            pending = path.with_suffix('.pending')
            LOGGER.info(
                'Requesting ERA5-Land %s%s: %s',
                month.strftime('%Y-%m'),
                ' boundary midnight' if boundary else '',
                ', '.join(variables),
            )
            self.client.retrieve('reanalysis-era5-land', request).download(str(pending))
            # Cache only readable/complete responses. Failed downloads are retried by
            # the next external job; credentials and CDS transport remain with cdsapi.
            ds = read_grib(pending, expected, required)
            _validate(ds, month, boundary=boundary, required=required)
            pending.replace(path)
            return read_grib(path, expected, required)
        LOGGER.info('Reusing ERA5-Land GRIB cache: %s', path.name)
        ds = read_grib(path, expected, required)
        _validate(ds, month, boundary=boundary, required=required)
        return ds


def _daily_batches(
    main: xr.Dataset, boundary: xr.Dataset, month: date, bbox: tuple, cache: DownloadCache
):
    """Keep the entire month uncommitted until all days and cache uploads succeed."""
    days = calendar.monthrange(month.year, month.month)[1]
    for offset in range(days):
        day = month + offset * const.ONE_DAY
        LOGGER.info('Reducing and writing ERA5-Land %s (%s/%s)', day, offset + 1, days)
        ds = process_day(main, boundary, day)
        ds.attrs['request_bbox'] = list(bbox)
        yield ds
        del ds
    cache.publish(cache.root.glob('*.grib'))


def process_day(main: xr.Dataset, boundary: xr.Dataset, day: date) -> xr.Dataset:
    """Reduce one spatial field at a time; memory is independent of month length.

    A full field is a natural GRIB decoding block, spanning multiple Zarr chunks
    (with partial chunks only at grid edges). The daily output writes each time
    chunk once. Shared hourly inputs are decoded once for all derived variables.
    """
    start = np.datetime64(day, 'h')
    following = start + np.timedelta64(24, 'h')

    last = main.time.to_numpy()[-1]

    def read(name: str, stamp: np.datetime64):
        source = main if stamp <= last else boundary
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
    return _prepare_daily_values(
        values, np.array([np.datetime64(day)]), main.latitude, main.longitude, main.rio.crs
    )


def _reduce_grib_field(read, spec: const.Era5Var, start: np.datetime64):  # noqa: ANN001
    reduced = read(spec.nc_name, start).copy()
    for hour in range(1, 24):
        field = read(spec.nc_name, start + np.timedelta64(hour, 'h'))
        if spec.daily_stat is np.mean:
            reduced += field
        elif spec.daily_stat is np.max:
            np.maximum(reduced, field, out=reduced)
        elif spec.daily_stat is np.min:
            np.minimum(reduced, field, out=reduced)
        else:
            msg = 'Instantaneous variable requires daily statistic'
            raise StorageError(msg)
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


def _validate(
    ds: xr.Dataset, month: date, *, boundary: bool, required: set[str] | None = None
) -> None:
    if required is None:
        required = {v.nc_name for v in const.ERA5_VARS.values()} | set(EXTRA)
    missing = required - {str(name) for name in ds.data_vars}
    if missing:
        msg = f'Missing ERA5 variables: {sorted(missing)}'
        raise StorageError(msg)
    start = np.datetime64(month, 'h')
    expected = (
        np.array([start])
        if boundary
        else np.arange(start, np.datetime64(next_month(month), 'h'), np.timedelta64(1, 'h'))
    )
    if 'time' not in ds.coords or not np.array_equal(ds.time.to_numpy(), expected):
        msg = f'Incomplete or unexpected ERA5 timestamps for {month}'
        raise StorageError(msg)
    for name in required:
        if set(ds[name].dims) != {'time', 'latitude', 'longitude'}:
            msg = f'Unexpected ERA5 dimensions for {name}: {ds[name].dims}'
            raise StorageError(msg)


def _validate_boundary(main: xr.Dataset, boundary: xr.Dataset) -> None:
    if main.rio.crs != boundary.rio.crs or any(
        not np.array_equal(main[axis], boundary[axis]) for axis in ('latitude', 'longitude')
    ):
        msg = 'ERA5 month and following midnight have different coordinates or CRS'
        raise StorageError(msg)


def _prepare_daily_values(values: dict, times: np.ndarray, latitudes, longitudes, crs):  # noqa: ANN001
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
        array = value.astype(np.float32, copy=False)
        if array.ndim == 2:
            array = array[np.newaxis]
        fields[name] = (('time', 'latitude', 'longitude'), array, attrs)
    ds = xr.Dataset(
        fields,
        coords={
            'time': times,
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )
    return prepare_dataset(ds, crs=crs)


def _validate_bbox(ds: xr.Dataset, bbox: tuple) -> None:
    north, west, south, east = bbox
    for axis, lower, upper in [('latitude', south, north), ('longitude', west, east)]:
        values = ds[axis].to_numpy()
        if axis == 'longitude':
            values = (values - west) % 360 + west
        # CDS rounds bbox limits to source cells. Global longitude omits the
        # duplicated east seam. Require coverage to within one source step.
        tolerance = const.ERA5_RESOLUTION + 1e-7
        if (
            abs(float(values.min()) - lower) > tolerance
            or abs(float(values.max()) - upper) > tolerance
        ):
            msg = f'ERA5 {axis} does not match the requested bbox'
            raise StorageError(msg)
