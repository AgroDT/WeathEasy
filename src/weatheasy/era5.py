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
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
    bbox: tuple[float, float, float, float] | None = None,
    max_pending_requests: int = 3,
) -> None:
    """Append complete months through the last month with available midnight data.

    bbox is (north, west, south, east), without antimeridian crossing. The CDS
    publication lag is five days; only whole months are requested. Run as an
    external job. No download is triggered by API or prepared-data reads.
    """
    if max_pending_requests < 1:
        msg = 'max_pending_requests must be at least 1'
        raise ValueError(msg)
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
        marker = [previous.year, previous.month]
        if group_attrs(store, '').get('last_month') != marker:
            # Recover a crash after committing daily data but before journal cleanup.
            set_group_attrs(store, '', last_month=marker)
    with download_cache(download_dir, 'era5', work_dir=work_dir, target=str(store)) as cache:
        downloader = _Downloader(None, cache, bbox, store, max_pending_requests)
        downloader.cleanup(first, latest, max_age_days=work_max_age_days)
        if next_month(first) > latest:
            set_group_attrs(store, '', era5_requests={})
            return
        try:
            # The CDS client is optional and needed only for a pending download.
            import cdsapi  # noqa: PLC0415
        except ImportError as err:
            msg = 'ERA5 preparation requires weatheasy[prepare]'
            raise ImportError(msg) from err
        # Keep cdsapi configuration/logging while using its DSS backend's job API.
        downloader.client = cast('Any', cdsapi.Client(wait_until_complete=False)).client
        while next_month(first) <= latest:
            following = next_month(first)
            started = time.monotonic()
            downloader.plan(first, latest)
            LOGGER.info('Preparing ERA5-Land %s', first.strftime('%Y-%m'))
            main, boundary = downloader.retrieve_month(first)
            try:
                _validate_boundary(main, boundary)
                write_batches(store, path, _daily_batches(main, boundary, first, bbox))
            finally:
                main.close()
                boundary.close()
            downloader.commit(first)
            LOGGER.info('Prepared ERA5-Land %s in %.1f s', first, time.monotonic() - started)
            first = following


@dataclass(frozen=True)
class _Part:
    month: date
    boundary: bool
    request: dict
    required: set[str]
    key: str

    @property
    def filename(self) -> str:
        return f'{self.month:%Y%m}-{self.key}.grib'


class _Downloader:
    __slots__ = ('bbox', 'cache', 'client', 'limit', 'parts', 'pending', 'requests', 'store')

    def __init__(
        self,
        client: Any,
        cache: DownloadCache | Path,
        bbox: tuple,
        store: Store,
        max_pending_requests: int = 3,
    ) -> None:
        self.client = client
        self.cache = DownloadCache(cache) if isinstance(cache, Path) else cache
        self.bbox = bbox
        self.store = store
        self.limit = max_pending_requests
        self.parts: dict[str, _Part] = {}
        self.pending: dict[str, Any] = {}
        self.requests: dict[str, str] = (
            group_attrs(store, '').get('era5_requests', {}) if group_exists(store, '') else {}
        )

    def _request(self, key: str, request: dict) -> Any:
        from requests import HTTPError  # noqa: PLC0415

        if request_id := self.requests.get(key):
            try:
                remote = self.client.get_remote(request_id)
            except HTTPError as exc:
                if exc.response is None or exc.response.status_code not in (404, 410):
                    raise
                LOGGER.info('ERA5-Land request %s is no longer available; resubmitting', request_id)
            else:
                if remote.status not in ('failed', 'rejected', 'dismissed', 'deleted'):
                    LOGGER.info('Resuming ERA5-Land request %s', request_id)
                    return remote
                LOGGER.info('ERA5-Land request %s has failed or expired; resubmitting', request_id)
        remote = self.client.submit('reanalysis-era5-land', request)
        self.requests[key] = remote.request_id
        # Persist before waiting/downloading, independently of raw-cache publication.
        set_group_attrs(self.store, '', era5_requests=self.requests)
        return remote

    def _parts(self, month: date, *, boundary: bool) -> list[_Part]:
        names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
        names.update({cds: nc for nc, cds in EXTRA.items()})
        variables = sorted(names)
        return [
            self._part(month, variables[offset : offset + 8], names, boundary=boundary)
            for offset in range(0, len(variables), 8)
        ]

    def cleanup(self, first: date, latest: date, *, max_age_days: int) -> None:
        """Protect every pending month and its boundary before age-based cleanup."""
        needed = set()
        month = first
        while next_month(month) <= latest:
            needed.update(
                part.filename
                for part in self._parts(month, boundary=False)
                + self._parts(next_month(month), boundary=True)
            )
            month = next_month(month)
        self.cache.cleanup(needed.__contains__, max_age_days=max_age_days)

    def plan(self, month: date, latest: date) -> None:
        # Only this month and the next may occupy the submission window.
        for owner in (month, next_month(month)):
            if next_month(owner) <= latest:
                for part in self._parts(owner, boundary=False) + self._parts(
                    next_month(owner), boundary=True
                ):
                    self.parts[part.key] = part
        self.requests = {key: value for key, value in self.requests.items() if key in self.parts}
        set_group_attrs(self.store, '', era5_requests=self.requests)
        self._fill()

    def _fill(self) -> None:
        for key, part in self.parts.items():
            if len(self.pending) >= self.limit:
                break
            if key not in self.pending and not self.cache.exists(part.filename):
                self.pending[key] = self._request(key, part.request)

    def retrieve_month(self, month: date) -> tuple[xr.Dataset, xr.Dataset]:
        main = self._parts(month, boundary=False)
        boundary = self._parts(next_month(month), boundary=True)
        pieces = self._retrieve(main + boundary)
        return (
            xr.merge(pieces[: len(main)], join='exact', compat='identical'),
            xr.merge(pieces[len(main) :], join='exact', compat='identical'),
        )

    def retrieve(self, month: date, *, boundary: bool) -> xr.Dataset:
        return xr.merge(
            self._retrieve(self._parts(month, boundary=boundary)), join='exact', compat='identical'
        )

    def _retrieve(self, parts: list[_Part]) -> list[xr.Dataset]:
        self.parts.update((part.key, part) for part in parts)
        self._fill()
        wanted = {part.key for part in parts if not self.cache.exists(part.filename)}
        pieces = {part.key: self._read(part) for part in parts if part.key not in wanted}
        while wanted:
            for key, remote in self.pending.items():
                if key in wanted and remote.results_ready:
                    part = self.parts[key]
                    path = self.cache.file(part.filename)
                    pending = path.with_name(path.name + '.part')
                    LOGGER.info('Downloading ERA5-Land GRIB: %s', path.name)
                    started = time.monotonic()
                    remote.download(str(pending))
                    LOGGER.info('Downloaded %s in %.1f s', path.name, time.monotonic() - started)
                    pending.replace(path)
                    del self.pending[key]
                    wanted.remove(key)
                    # Let CDS work while this file is uploaded and processed.
                    self._fill()
                    started = time.monotonic()
                    self.cache.publish([path])
                    LOGGER.info('Cached %s in %.1f s', path.name, time.monotonic() - started)
                    pieces[key] = self._read(part)
                    break
            else:
                time.sleep(10)
        return [pieces[part.key] for part in parts]

    def commit(self, month: date) -> None:
        parts = self._parts(month, boundary=False) + self._parts(next_month(month), boundary=True)
        for part in parts:
            self.requests.pop(part.key, None)
            self.parts.pop(part.key, None)
        set_group_attrs(
            self.store, '', last_month=[month.year, month.month], era5_requests=self.requests
        )
        self.cache.release(self.cache.root / part.filename for part in parts)

    def _part(
        self,
        month: date,
        variables: list[str],
        names: dict[str, str],
        *,
        boundary: bool,
    ) -> _Part:
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
        return _Part(month, boundary, request, {names[name] for name in variables}, key)

    def _read(self, part: _Part) -> xr.Dataset:
        path = self.cache.file(part.filename)
        # Also retry a publication that failed after a complete local download.
        self.cache.publish([path])
        return read_grib(path, part.required)


def _daily_batches(main: xr.Dataset, boundary: xr.Dataset, month: date, bbox: tuple):
    """Keep the entire month uncommitted until all days are processed."""
    days = calendar.monthrange(month.year, month.month)[1]
    for offset in range(days):
        day = month + offset * const.ONE_DAY
        LOGGER.info('Reducing and writing ERA5-Land %s (%s/%s)', day, offset + 1, days)
        ds = process_day(main, boundary, day)
        ds.attrs['request_bbox'] = list(bbox)
        yield ds
        del ds


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
    result = _prepare_daily_values(
        values, np.array([np.datetime64(day)]), main.latitude, main.longitude, main.rio.crs
    )
    source_grib = {
        label: {
            name: {'edition': field.attrs['GRIB_edition'], 'radius': field.attrs['GRIB_radius']}
            for name, field in source.data_vars.items()
            if 'GRIB_radius' in field.attrs
        }
        for label, source in (('main', main), ('boundary', boundary))
    }
    if any(source_grib.values()):
        result.attrs['source_grib'] = source_grib
    return result


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
