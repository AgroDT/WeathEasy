from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack
from datetime import date
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from httpx2 import Client, HTTPTransport, ReadError, ReadTimeout, RemoteProtocolError

from weatheasy import const
from weatheasy.raw_cache import DownloadCache, default_work_dir, download_cache
from weatheasy.sources import read_cfs2, read_netcdf
from weatheasy.storage import (
    delete_group,
    group_attrs,
    group_exists,
    last_date,
    prepare_dataset,
    recover_group,
    remove_group_attr,
    replace_blocks,
    set_group_attrs,
    write_batches,
    write_blocks,
)
from weatheasy.util import utc_now


if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import BinaryIO, Literal

    from httpx2 import Response
    from zarr.abc.store import Store

    from weatheasy.file_storage import FileStore


def download_cfs2_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
    proxy: str | None = None,
    reanalysis_recheck_days: int = 30,
) -> None:
    if reanalysis_recheck_days < 0:
        msg = 'reanalysis_recheck_days must be non-negative'
        raise ValueError(msg)
    work_dir = Path(work_dir).expanduser() if work_dir is not None else default_work_dir()
    set_group_attrs(store, '')
    today = utc_now().date()
    updated = group_attrs(store, '').get(const.CFS2_KEY_UPDATED)
    pending = not updated or today - date.fromisoformat(str(updated)) >= const.ONE_DAY
    yesterday = today - const.ONE_DAY
    forecast_begin = today - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
    forecast_end = yesterday + const.CFS2_FORECAST_DAYS
    for kind in ('flx', 'pgb'):
        path = f'reanalysis/{kind}'
        if group_exists(store, path):
            recover_group(store, path)
    retry_days = _cfs2_retry_days(
        store, today - reanalysis_recheck_days * const.ONE_DAY, forecast_begin
    )
    retries = set().union(*retry_days.values())

    with ExitStack() as stack:
        reanalysis = stack.enter_context(
            download_cache(
                download_dir,
                'cfs2',
                const.CFS2_REANALYSIS_DIR,
                work_dir=work_dir,
                target=str(store),
            )
        )
        forecast_cache = stack.enter_context(
            download_cache(
                download_dir, 'cfs2', const.CFS2_FORECAST_DIR, work_dir=work_dir, target=str(store)
            )
        )
        previous = [last_date(store, f'reanalysis/{kind}') for kind in ('flx', 'pgb')]
        first = min(
            value + const.ONE_DAY if value else const.CFS2_REANALYSIS_FIRST_DATE
            for value in previous
        )

        def needed_reanalysis(name: str) -> bool:
            try:
                day = date.fromisoformat(name.rsplit('/', 1)[-1].split('.', maxsplit=1)[0])
            except ValueError:
                return False
            return (pending and first <= day < forecast_begin) or day in retries

        reanalysis.cleanup(needed_reanalysis, max_age_days=work_max_age_days)
        forecast_names = (
            {
                f'{today.isoformat()}/{kind}{day}{hour}.grb2'
                for kind, day, hour in product(
                    ('flx', 'pgb'),
                    _cfs2_forecast_dates(forecast_begin, forecast_end),
                    const.CFS2_HHS,
                )
            }
            if pending
            else set()
        )
        forecast_cache.cleanup(forecast_names.__contains__, max_age_days=work_max_age_days)
        if not pending and not retries:
            _LOG.info('No forecast update required')
            return
        session = stack.enter_context(_session(proxy))
        _download_cfs2_reanalysis(
            store, forecast_begin, session, reanalysis, work_dir=work_dir, retry_days=retry_days
        )
        if not pending:
            _LOG.info('No forecast update required')
            return
        forecast = forecast_cache.child(today.isoformat())
        _download_cfs2_forecast(forecast_begin, yesterday, session, forecast)
        _download_cfs2_forecast(yesterday, forecast_end, session, forecast)
        _merge_cfs2_forecast(store, forecast_begin, forecast_end, forecast.root, work_dir=work_dir)
        set_group_attrs(store, '', updated=today.isoformat())
        forecast.release(forecast.root.glob('*.grb2'))


def download_cmip6_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
    proxy: str | None = None,
) -> None:
    set_group_attrs(store, '')
    with download_cache(
        download_dir, const.CMIP6_DIR, work_dir=work_dir, target=str(store)
    ) as cache:
        first_years = {}
        for var in const.CMIP6_VARS:
            previous = last_date(store, var)
            first_years[var] = previous.year + 1 if previous else const.CMIP6_FIRST_YEAR
        needed = {
            f'{var}_{year}.nc'
            for var, first in first_years.items()
            for year in range(first, const.CMIP6_LAST_YEAR + 1)
        }
        cache.cleanup(needed.__contains__, max_age_days=work_max_age_days)
        with _session(proxy) as session:
            for var, first in first_years.items():
                for year in range(first, const.CMIP6_LAST_YEAR + 1):
                    source = _load_cmip6_dataset(cache, var, year, session)
                    dataset = read_netcdf(
                        source, units={var: const.CMIP6_UNITS[var]}, crs=const.CMIP6_CRS
                    )
                    # Normalize provider noon labels to calendar days.
                    dataset = dataset.assign_coords(
                        time=xr.DataArray(
                            dataset.time.to_numpy()
                            .astype('datetime64[D]')
                            .astype('datetime64[ns]'),
                            dims='time',
                            attrs=dataset.time.attrs,
                        )
                    )
                    _LOG.info('Saving %s/%s', var, year)
                    write_batches(
                        store,
                        var,
                        (
                            dataset.isel(time=slice(offset, offset + 128))
                            for offset in range(0, dataset.sizes['time'], 128)
                        ),
                    )
                    cache.release([source])


_MODULE_NAME = 'weatheasy.download'
_LAST = 'last'
_LOG = logging.getLogger(_MODULE_NAME)
_CFS2_DOWNLOADED_FILENAME_TEMPLATE = '{}{}{}.grb2'


def _session(proxy: str | None = None) -> Client:
    return Client(transport=HTTPTransport(retries=3, proxy=proxy), follow_redirects=True)


def _download_cfs2_reanalysis(
    store: Store,
    end: date,
    session: Client,
    download_dir: DownloadCache,
    *,
    work_dir: Path,
    retry_days: dict[str, list[date]] | None = None,
):
    download = _Cfs2ReanalysisDownloader(session, download_dir)
    groups = {kind: f'{const.CFS2_REANALYSIS_DIR}/{kind}' for kind in ('flx', 'pgb')}
    for kind in groups:
        set_group_attrs(store, f'{const.CFS2_REANALYSIS_DIR}/{kind}')
    previous = {kind: last_date(store, path) for kind, path in groups.items()}
    date_ = min(
        last + const.ONE_DAY if last else const.CFS2_REANALYSIS_FIRST_DATE
        for last in previous.values()
    )
    if date_ >= end:
        _repair_cfs2_reanalysis(store, retry_days or {}, download, download_dir, work_dir=work_dir)
        return
    templates, prefetched = _reanalysis_templates(
        store, groups, previous, date_, end, download=download
    )
    while date_ < end:
        block_end = min(date_ + 128 * const.ONE_DAY, end)
        paths_by_day = []
        while date_ < block_end:
            paths = prefetched.pop(date_) if date_ in prefetched else download(date_) or []
            paths_by_day.append((date_, paths))
            if len(paths) < len(const.CFS2_HHS):
                _LOG.warning(
                    'CFS2 %s: found %s of %s files; saving NaN for all fields',
                    date_,
                    len(paths),
                    len(const.CFS2_HHS),
                )
            date_ += const.ONE_DAY
        for kind, path in groups.items():
            last = previous[kind]
            _cfs2_incomplete_days(
                store,
                path,
                add=[
                    day
                    for day, paths in paths_by_day
                    if (last is None or day > last) and len(paths) < len(const.CFS2_HHS)
                ],
            )
            _LOG.info(
                'Processing %s: %s calendar days; writing in 128-day blocks',
                path,
                len(paths_by_day),
            )
            write_blocks(
                store,
                path,
                (
                    _daily_cfs2(paths, kind, day, templates[kind])
                    for day, paths in paths_by_day
                    if last is None or day > last
                ),
                work_dir=work_dir,
            )
            previous[kind] = last_date(store, path)
            _cfs2_incomplete_days(
                store,
                path,
                remove=[
                    day
                    for day, paths in paths_by_day
                    if (last is None or day > last) and len(paths) == len(const.CFS2_HHS)
                ],
            )
            _LOG.info('Saved %s through %s', path, previous[kind])
        for _, paths in paths_by_day:
            if len(paths) == len(const.CFS2_HHS):
                download_dir.release(paths)
    if retry_days:
        _repair_cfs2_reanalysis(store, retry_days, download, download_dir, work_dir=work_dir)


def _cfs2_incomplete_days(
    store: Store, path: str, *, add: Iterable[date] = (), remove: Iterable[date] = ()
) -> None:
    kind = path.rsplit('/', 1)[-1]
    tracked = dict(group_attrs(store, '').get('cfs2_incomplete_days', {}))
    before = tracked.get(kind, [])
    days = sorted((set(before) | {str(day) for day in add}) - {str(day) for day in remove})
    if days != before:
        tracked[kind] = days
        set_group_attrs(store, '', cfs2_incomplete_days=tracked)


def _cfs2_retry_days(store: Store, begin: date, end: date) -> dict[str, list[date]]:
    result = {'flx': [], 'pgb': []}
    begin = max(begin, const.CFS2_REANALYSIS_FIRST_DATE)
    if begin >= end:
        return result
    tracked = group_attrs(store, '').get('cfs2_incomplete_days', {})
    for kind in ('flx', 'pgb'):
        path = f'reanalysis/{kind}'
        if group_exists(store, path) and (last := last_date(store, path)) is not None:
            result[kind] = sorted(
                day
                for value in tracked.get(kind, [])
                if begin <= (day := date.fromisoformat(value)) < end and day <= last
            )
            _LOG.info('CFS2 %s: %s incomplete days in recheck window', kind, len(result[kind]))
    return result


def _repair_cfs2_reanalysis(
    store: Store,
    retry_days: dict[str, list[date]],
    download: _Cfs2ReanalysisDownloader,
    download_dir: DownloadCache,
    *,
    work_dir: Path,
) -> None:
    available = {}
    for day in sorted(set().union(*retry_days.values())):
        paths = download(day)
        if len(paths) == len(const.CFS2_HHS):
            available[day] = paths
        else:
            _LOG.info('CFS2 %s: still incomplete (%s/4 files)', day, len(paths))
    for kind, days in retry_days.items():
        ready = [day for day in days if day in available]
        if not ready:
            continue
        # Raw changes invalidate an already published correction before mutation.
        if kind == 'flx' and group_exists(store, 'reanalysis/correction'):
            attrs = group_attrs(store, 'reanalysis/correction')
            if attrs.get('committed_length'):
                earliest = min(str(ready[0]), str(attrs.get('source_changed_from', ready[0])))
                set_group_attrs(
                    store,
                    'reanalysis/correction',
                    source_changed_from=earliest,
                    write_state='writing',
                )
        replace_blocks(
            store,
            f'reanalysis/{kind}',
            (_daily_cfs2(available[day], kind, day) for day in ready),
            work_dir=work_dir,
        )
        _cfs2_incomplete_days(store, f'reanalysis/{kind}', remove=ready)
        _LOG.info('CFS2 %s: repaired %s previously incomplete days', kind, len(ready))
    for paths in available.values():
        download_dir.release(paths)


def _reanalysis_templates(
    store: Store,
    groups: dict[str, str],
    previous: dict[str, date | None],
    begin: date,
    end: date,
    *,
    download: _Cfs2ReanalysisDownloader,
):
    templates = {}
    for kind, path in groups.items():
        if previous[kind] is not None:
            # Only the grid/field metadata are needed, also after an interrupted
            # append. Never read possibly uncommitted weather values here.
            with xr.open_zarr(
                store, group=path, consolidated=False, chunks=None, decode_coords='all'
            ) as saved:
                templates[kind] = xr.full_like(saved.isel(time=slice(0, 1)), np.nan).load()
    # A new store may start with missing days. Find a real source grid before
    # writing them, retaining the downloaded paths for the normal block loop.
    prefetched = {}
    sample_day = begin
    while len(templates) < len(groups) and sample_day < end:
        paths = download(sample_day) or []
        prefetched[sample_day] = paths
        if paths:
            for kind in groups:
                if kind not in templates:
                    templates[kind] = read_cfs2(paths[0], kind=kind)
        sample_day += const.ONE_DAY
    if len(templates) < len(groups):
        msg = 'Cannot determine the reanalysis grid without any source files'
        raise ValueError(msg)
    return templates, prefetched


def _daily_cfs2(
    paths: list[Path],
    kind: str,
    date_: date,
    template: xr.Dataset | None = None,
    *,
    product: Literal['reanalysis', 'forecast'] = 'reanalysis',
):
    started = time.perf_counter()
    _LOG.info('Processing CFS2 %s %s: %s GRIB files', kind, date_, len(paths))
    fields = [read_cfs2(path, kind=kind, product=product) for path in paths]
    bands = const.CFS2_FLX_BANDS if kind == 'flx' else const.CFS2_PGB_BANDS
    if len(paths) < len(const.CFS2_HHS):
        if fields:
            template = fields[0]
        if template is None:
            msg = 'Cannot determine the grid without any source files'
            raise ValueError(msg)
        daily = xr.full_like(template, np.nan)
        for name, info in bands.items():
            daily[name].attrs['cell_methods'] = f'time: {info.daily_stat.__name__}'
    else:
        sample = fields[0]
        # Exact spatial alignment; only the sample time axis may differ.
        combined = xr.concat(fields, dim='time', join='exact')
        data = {}
        for name, info in bands.items():
            values = combined[name].to_numpy()
            data[name] = xr.DataArray(
                info.daily_stat(values, axis=0),
                dims=('latitude', 'longitude'),
                attrs={**sample[name].attrs, 'cell_methods': f'time: {info.daily_stat.__name__}'},
            )
        daily = xr.Dataset(data, coords={axis: sample[axis] for axis in ('latitude', 'longitude')})
    daily = daily.drop_vars('spatial_ref', errors='ignore')
    if 'time' in daily.dims:
        daily = daily.isel(time=0, drop=True)
    daily = daily.expand_dims(time=[np.datetime64(date_)])
    result = prepare_dataset(daily, crs=const.CFS2_CRS)
    source_grib = {
        str(field.time.to_numpy()[0]): field.attrs['source_grib']
        for field in fields
        if 'source_grib' in field.attrs
    }
    if source_grib:
        result.attrs['source_grib'] = source_grib
    _LOG.info('Processed CFS2 %s %s in %.2f s', kind, date_, time.perf_counter() - started)
    return result


def _download_cfs2_forecast(
    begin: date,
    end: date,
    session: Client,
    download_dir: DownloadCache,
):
    download = _Cfs2ForecastDownloader(session, download_dir, begin, end)
    download('flx', f'flxf{{}}{{}}.01.{begin:%Y%m%d}00.grb2', const.CFS2_FLX_PARAMS)
    download('pgb', f'pgbf{{}}{{}}.01.{begin:%Y%m%d}00.grb2', const.CFS2_PGB_PARAMS)


def _merge_cfs2_forecast(
    store: Store, begin: date, end: date, download_dir: Path, *, work_dir: Path
):
    # In-place replacement deliberately makes forecast data unavailable. Clear
    # readiness before deleting any chunks, then publish only after both grids.
    remove_group_attr(store, '', const.CFS2_KEY_UPDATED)
    delete_group(store, const.CFS2_FORECAST_DIR)
    for kind in ('flx', 'pgb'):
        group = f'{const.CFS2_FORECAST_DIR}/{kind}'
        set_group_attrs(store, group)
        template = None
        paths_by_day = []
        date_ = begin
        while date_ < end:
            paths = [
                download_dir
                / _CFS2_DOWNLOADED_FILENAME_TEMPLATE.format(kind, date_.strftime('%Y%m%d'), hour)
                for hour in const.CFS2_HHS
            ]
            paths = [path for path in paths if path.is_file()]
            paths_by_day.append((date_, paths))
            if template is None and paths:
                template = read_cfs2(paths[0], kind=kind, product='forecast')
            date_ += const.ONE_DAY
        if template is None:
            msg = f'No forecast files for {kind}; forecast remains unavailable'
            raise ValueError(msg)
        _LOG.info('Processing %s: %s days; writing in 128-day blocks', group, len(paths_by_day))
        write_blocks(
            store,
            group,
            (
                _daily_cfs2(paths, kind, date_, template, product='forecast')
                for date_, paths in paths_by_day
            ),
            work_dir=work_dir,
            time_chunk=64,
        )
        _LOG.info('Saved %s: %s days', group, len(paths_by_day))


def _complete_grib2(stream: BinaryIO) -> bool:
    """Check every message envelope without decoding weather fields or writing indexes."""
    size = stream.seek(0, 2)
    offset = 0
    while offset < size:
        stream.seek(offset)
        header = stream.read(16)
        if len(header) != 16 or header[:4] != b'GRIB' or header[7] != 2:
            return False
        length = int.from_bytes(header[8:16], 'big')
        if length < 20 or length > size - offset:
            return False
        stream.seek(offset + length - 4)
        if stream.read(4) != b'7777':
            return False
        offset += length
    return size > 0


class _Cfs2ReanalysisDownloader:
    def __init__(self, session: Client, download_dir: DownloadCache) -> None:
        self._session = session
        self._download_dir = download_dir

    def _get(self, url: str) -> Response:
        # Transport retries cover connection setup, not interrupted response bodies.
        for attempt in range(3):
            try:
                return self._session.get(url, timeout=180)
            except (ReadError, ReadTimeout, RemoteProtocolError):
                delay = 2**attempt
                _LOG.warning(
                    'CFS2 download interrupted (attempt %s/4); retrying in %s s: %s',
                    attempt + 1,
                    delay,
                    url,
                )
                time.sleep(delay)
        return self._session.get(url, timeout=180)

    def __call__(self, date_: date) -> list[Path]:
        ymd = date_.strftime('%Y%m%d')
        ym = ymd[:-2]
        url_template = (
            'https://www.ncei.noaa.gov/oa/prod-cfs-operational-analysis/6-hourly-by-pressure/'
            f'{date_.year}/{ym}/{ymd}/cdas1.t{{}}z.pgrbh00.grib2'
        )
        paths = []
        for hhs in const.CFS2_HHS:
            path = self._download_dir.file(f'{ym}/{ymd}.cdas1.t{hhs}z.pgrbh00.grib2')
            invalid_cached = False
            if path.is_file():
                with path.open('rb') as stream:
                    invalid_cached = not _complete_grib2(stream)
            if invalid_cached:
                _LOG.warning('Retrying incomplete or invalid CFS2 cache file: %s', path)
                path.unlink()
            if not path.is_file():
                url = url_template.format(hhs)
                _LOG.info('Downloading %s', url)
                response = self._get(url)
                if response.status_code == 404:
                    _LOG.warning('Missing CFS2 file: %s', url)
                    continue
                if not response.is_success:
                    msg = f'Failed to download {url}'
                    raise RuntimeError(msg)
                if not response.content:
                    _LOG.warning('Empty CFS2 response; treating file as missing: %s', url)
                    continue
                if not _complete_grib2(BytesIO(response.content)):
                    msg = f'Incomplete or invalid CFS2 GRIB download: {url} (cache file: {path})'
                    raise RuntimeError(msg)
                _LOG.info('Writing %s', path)
                self._download_dir.write(path, response.content)
            self._download_dir.publish([path], overwrite=invalid_cached)
            paths.append(path)
        return paths


class _Cfs2ForecastDownloader:
    _base_url: ClassVar = 'https://nomads.ncep.noaa.gov/cgi-bin/'
    _min_interval: ClassVar = 1.0 / 3.0

    def __init__(
        self, session: Client, download_dir: DownloadCache, begin: date, end: date
    ) -> None:
        self._session = session
        self._download_dir = download_dir
        self._begin = begin
        self._end = end
        self._dir = f'/cfs.{begin:%Y%m%d}/00/6hrly_grib_01'
        self._last_call = 0.0

    def __call__(self, kind: str, filename_tmpl: str, params: dict) -> None:
        url = self._base_url + f'filter_cfs_{kind}.pl'
        params = {**params, 'dir': self._dir}
        for date_str, hhs in product(_cfs2_forecast_dates(self._begin, self._end), const.CFS2_HHS):
            file = filename_tmpl.format(date_str, hhs)
            path = self._download_dir.file(
                _CFS2_DOWNLOADED_FILENAME_TEMPLATE.format(kind, date_str, hhs)
            )
            if path.is_file():
                self._download_dir.publish([path])
                _LOG.info('Skipping already downloaded %s', path)
                continue
            elapsed = time.time() - self._last_call
            delta = self._min_interval - elapsed
            if delta > 0:
                time.sleep(delta)
            self._last_call = time.time()
            _LOG.info('Downloading %s', file)
            params['file'] = file
            response = self._session.get(url, params=params, timeout=180)
            if response.is_success:
                if response.content.startswith(b'<!doctype html>'):
                    _LOG.critical('Exceeded overrate limit for nomads.ncep.noaa.gov')
                    sys.exit(1)
                _LOG.info('Writing %s', path)
                self._download_dir.write(path, response.content)
                self._download_dir.publish([path])
            else:
                _LOG.critical('Failed to download %s', response.url)


def _cfs2_forecast_dates(date_: date, end: date):
    while date_ < end:
        yield date_.strftime('%Y%m%d')
        date_ += const.ONE_DAY


def _load_cmip6_dataset(cache: DownloadCache, var: str, year: int, session: Client) -> Path:
    filename = f'{var}_{year}.nc'
    path = cache.file(filename)
    if path.is_file():
        cache.publish([path])
        _LOG.info('Reading %s', path)
        return path

    kind = 'historical' if year <= const.CMIP6_LAST_HISTORICAL_YEAR else 'ssp245'
    url = (
        'https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/ACCESS-CM2/'
        f'{kind}/r1i1p1f1/{var}/{var}_day_ACCESS-CM2_{kind}_r1i1p1f1_gn_{year}.nc'
    )
    _LOG.info('Downloading %s', url)
    response = session.get(url, timeout=180)
    if not response.is_success:
        _LOG.critical('Failed to download %s', url)
        sys.exit(1)
    _LOG.info('Writing %s', path)
    cache.write(path, response.content)
    cache.publish([path])
    return path
