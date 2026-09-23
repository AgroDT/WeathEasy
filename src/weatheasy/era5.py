"""ERA5-Land monthly preparation jobs; hourly inputs are loaded eagerly.

Accumulated fields use the following midnight, including at month/year boundaries.
PEVPR is open-water evaporation, not FAO reference ET0. Evaporation and turbulent
heat fluxes are positive upwards; source signs are reversed without clipping.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from weatheasy import const
from weatheasy.storage import StorageError, last_date, prepare_dataset, write_batch
from weatheasy.util import utc_now


if TYPE_CHECKING:
    import zarr

EXTRA = {
    'd2m': '2m_dewpoint_temperature',
    'ssr': 'surface_net_solar_radiation',
    'str': 'surface_net_thermal_radiation',
}
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


def available_period(root: zarr.Group) -> tuple[date, date]:
    """Return the committed time range, including incomplete years.

    last_month is a convenience marker. A crash after committing data but before
    updating that marker must neither hide data nor cause a duplicate append.
    """
    path = '/'.join(filter(None, (root.path, const.ERA5_DIR, 'daily')))
    from weatheasy.storage import open_dataset

    with open_dataset(root.store, path) as ds:
        times = ds.time.to_numpy().astype('datetime64[D]')
        first = date.fromisoformat(str(times[0]))
        last = date.fromisoformat(str(times[-1]))
    return first, last


def download_era5_data(
    root: zarr.Group,
    download_dir: Path | None = None,
    *,
    area: tuple[float, float, float, float] | None = None,
) -> None:
    """Append complete months through the last month with available midnight data.

    area is (north, west, south, east), without antimeridian crossing. The CDS
    publication lag is five days; only whole months are requested. Run as an
    external job. No download is triggered by API or prepared-data reads.
    """
    area = area or (90.0, -180.0, -90.0, 180.0)
    north, west, south, east = area
    if not np.isfinite(area).all() or not (
        -90 <= south < north <= 90 and -180 <= west < east <= 180
    ):
        msg = 'Invalid ERA5 area (north, west, south, east)'
        raise StorageError(msg)
    if const.ERA5_DIR in root:
        existing = root[const.ERA5_DIR]
        if existing.attrs.get('request_area') != list(area):
            msg = 'Existing ERA5 store has a different requested area'
            raise StorageError(msg)
    group = root.require_group(const.ERA5_DIR)
    group.attrs['request_area'] = list(area)
    previous = last_date(root.store, group.path + '/daily')
    first = next_month(previous) if previous else date(const.ERA5_FIRST_YEAR, 1, 1)
    latest = utc_now().date() - 5 * const.ONE_DAY
    if previous:
        group.attrs['last_month'] = [previous.year, previous.month]
    if next_month(first) > latest:
        return
    try:
        import cdsapi
    except ImportError as err:
        msg = 'ERA5 preparation requires weatheasy[prepare]'
        raise ImportError(msg) from err
    client = cdsapi.Client()
    with TemporaryDirectory() as temporary:
        cache = (download_dir / 'era5') if download_dir else Path(temporary)
        cache.mkdir(parents=True, exist_ok=True)
        while next_month(first) <= latest:
            following = next_month(first)
            main = _retrieve(client, cache, first, area, boundary=False)
            boundary = _retrieve(client, cache, following, area, boundary=True)
            _validate_area(main, area)
            ds = process_month(main, boundary, first)
            # write_batch checks exact normalized axes, CRS, fields and units
            # before changing any existing arrays, even for equal-sized regions.
            ds.attrs['request_area'] = list(area)
            write_batch(root.store, group.path + '/daily', ds)
            group.attrs['last_month'] = [first.year, first.month]
            first = following


def _retrieve(client: Any, cache: Path, month: date, area: tuple, *, boundary: bool) -> xr.Dataset:
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
    names.update({cds: nc for nc, cds in EXTRA.items()})
    variables = sorted(names)
    pieces = []
    # CDS NetCDF costing counts two items per hourly field. Eight variables
    # in the longest month cost 8 * 31 * 24 * 2 = 11904, below the 12000 limit.
    for offset in range(0, len(variables), 8):
        subset = variables[offset : offset + 8]
        pieces.append(
            _retrieve_part(
                client,
                cache,
                month,
                area,
                subset,
                {names[name] for name in subset},
                boundary=boundary,
            )
        )
    ds = xr.merge(pieces, join='exact', compat='identical')
    _validate(ds, month, boundary=boundary)
    return ds


def _retrieve_part(
    client: Any,
    cache: Path,
    month: date,
    area: tuple,
    variables: list[str],
    required: set[str],
    *,
    boundary: bool,
) -> xr.Dataset:
    request = {
        'variable': variables,
        'year': str(month.year),
        'month': f'{month.month:02d}',
        'day': ['01']
        if boundary
        else [f'{d:02d}' for d in range(1, calendar.monthrange(month.year, month.month)[1] + 1)],
        'time': ['00:00'] if boundary else [f'{h:02d}:00' for h in range(24)],
        'area': list(area),
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    path = cache / f'{month:%Y%m}-{key}.nc'
    if not path.exists():
        pending = path.with_suffix('.pending')
        client.retrieve('reanalysis-era5-land', request).download(str(pending))
        # Cache only readable/complete responses. Failed downloads are retried by
        # the next external job; credentials and CDS transport remain with cdsapi.
        ds = _read_response(pending)
        _validate(ds, month, boundary=boundary, required=required)
        pending.replace(path)
        return ds
    ds = _read_response(path)
    _validate(ds, month, boundary=boundary, required=required)
    return ds


def _read_response(path: Path) -> xr.Dataset:
    def read(source: Path):
        with xr.open_dataset(source, engine='netcdf4', decode_coords='all') as raw:
            ds = raw.rename({'valid_time': 'time'}) if 'valid_time' in raw.dims else raw
            return ds.load()

    # CDS may split NetCDF by GRIB stepType, even for an unarchived request.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive, TemporaryDirectory() as temporary:
            pieces = []
            for index, name in enumerate(archive.namelist()):
                if name.endswith('.nc'):
                    target = Path(temporary) / f'{index}.nc'
                    target.write_bytes(archive.read(name))
                    pieces.append(read(target))
            if not pieces:
                msg = 'CDS archive contains no NetCDF data'
                raise StorageError(msg)
            return xr.merge(pieces, join='exact', compat='no_conflicts')
    return read(path)


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


def process_month(main: xr.Dataset, boundary: xr.Dataset, month: date) -> xr.Dataset:
    """Compute daily data, retaining NaNs and actual source coordinate centres."""
    _validate(main, month, boundary=False)
    _validate(boundary, next_month(month), boundary=True)
    for axis in ('latitude', 'longitude'):
        if not np.array_equal(main[axis], boundary[axis]):
            msg = 'ERA5 month and following midnight have different coordinates'
            raise StorageError(msg)
    days = calendar.monthrange(month.year, month.month)[1]
    shape = (days, 24, main.sizes['latitude'], main.sizes['longitude'])
    hourly = {
        name: main[name]
        .transpose('time', 'latitude', 'longitude')
        .to_numpy()
        .astype(np.float64)
        .reshape(shape)
        for name in main.data_vars
        if set(main[name].dims) == {'time', 'latitude', 'longitude'}
    }

    def accumulated(name: str):
        return np.concatenate(
            (
                hourly[name][1:, 0],
                boundary[name].transpose('time', 'latitude', 'longitude').to_numpy(),
            ),
            axis=0,
        )

    values = {}
    for name, spec in const.ERA5_VARS.items():
        reduced = (
            accumulated(spec.nc_name) if spec.accumulated else _reduce(spec, hourly[spec.nc_name])
        )
        values[name] = reduced * spec.scale + spec.offset
    t = hourly['t2m'] - 273.15
    td = hourly['d2m'] - 273.15
    es = 611.21 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))
    ea = 611.21 * np.exp((18.678 - td / 234.5) * td / (257.14 + td))
    q = 0.622 * ea / np.maximum(hourly['sp'] - 0.378 * ea, 1.0)
    values.update(
        RH=np.mean(np.clip(100 * ea / np.maximum(es, 1e-8), 0, 100), axis=1),
        SPFH=np.mean(q, axis=1),
        QMIN=np.min(q, axis=1),
        QMAX=np.max(q, axis=1),
        WIND=np.mean(np.hypot(hourly['u10'], hourly['v10']), axis=1),
        USWRF=(accumulated('ssrd') - accumulated('ssr')) / 86400,
        ULWRF=(accumulated('strd') - accumulated('str')) / 86400,
    )
    fields = {}
    for name, value in values.items():
        attrs = {'units': UNITS[name], 'long_name': const.ERA5_ALL_VARS[name].en}
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
        fields[name] = (('time', 'latitude', 'longitude'), value.astype(np.float32), attrs)
    ds = xr.Dataset(
        fields,
        coords={
            'time': np.arange(np.datetime64(month), np.datetime64(next_month(month))),
            'latitude': main.latitude,
            'longitude': main.longitude,
        },
    )
    # ECMWF recommends the GRIB1 sphere when converted NetCDF omits CRS.
    # Preserve explicit CF metadata when present; never relabel as WGS84.
    crs = main.rio.crs or '+proj=longlat +R=6367470 +no_defs'
    if boundary.rio.crs != main.rio.crs:
        msg = 'ERA5 month and following midnight have different CRS'
        raise StorageError(msg)
    return prepare_dataset(ds, crs=crs)


def _reduce(spec: const.Era5Var, values: np.ndarray) -> np.ndarray:
    if spec.daily_stat is None:
        msg = 'Instantaneous variable requires daily statistic'
        raise StorageError(msg)
    return spec.daily_stat(values, axis=1)


def _validate_area(ds: xr.Dataset, area: tuple) -> None:
    north, west, south, east = area
    for axis, lower, upper in [('latitude', south, north), ('longitude', west, east)]:
        values = ds[axis].to_numpy()
        if axis == 'longitude':
            values = (values - west) % 360 + west
        # CDS rounds area limits to source cells. Global longitude omits the
        # duplicated east seam. Require coverage to within one source step.
        tolerance = const.ERA5_RESOLUTION + 1e-7
        if (
            abs(float(values.min()) - lower) > tolerance
            or abs(float(values.max()) - upper) > tolerance
        ):
            msg = f'ERA5 {axis} does not match the requested area'
            raise StorageError(msg)
