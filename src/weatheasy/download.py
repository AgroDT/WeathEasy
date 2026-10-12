from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack
from datetime import date
from itertools import product
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from httpx2 import Client, HTTPTransport

from weatheasy import const
from weatheasy.raw_cache import DownloadCache, download_cache
from weatheasy.sources import read_cfs2, read_netcdf
from weatheasy.storage import (
    delete_group,
    group_attrs,
    last_date,
    prepare_dataset,
    remove_group_attr,
    set_group_attrs,
    write_batch,
)
from weatheasy.util import utc_now


if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import Store

    from weatheasy.file_storage import FileStore


def download_cfs2_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
) -> None:
    set_group_attrs(store, '')
    today = utc_now().date()
    updated = group_attrs(store, '').get(const.CFS2_KEY_UPDATED)
    pending = not updated or today - date.fromisoformat(str(updated)) >= const.ONE_DAY
    yesterday = today - const.ONE_DAY
    forecast_begin = today - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
    forecast_end = yesterday + const.CFS2_FORECAST_DAYS

    with ExitStack() as stack:
        reanalysis = stack.enter_context(
            download_cache(
                download_dir, const.CFS2_REANALYSIS_DIR, work_dir=work_dir, target=str(store)
            )
        )
        forecast_cache = stack.enter_context(
            download_cache(
                download_dir, const.CFS2_FORECAST_DIR, work_dir=work_dir, target=str(store)
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
            return pending and first <= day < forecast_begin

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
        if not pending:
            _LOG.info('No forecast update required')
            return
        forecast = forecast_cache.child(today.isoformat())
        session = stack.enter_context(_session())
        _download_cfs2_reanalysis(store, forecast_begin, session, reanalysis)
        _download_cfs2_forecast(forecast_begin, yesterday, session, forecast)
        _download_cfs2_forecast(yesterday, forecast_end, session, forecast)
        _merge_cfs2_forecast(store, forecast_begin, forecast_end, forecast.root)
        set_group_attrs(store, '', updated=today.isoformat())
        forecast.release(forecast.root.glob('*.grb2'))


def download_cmip6_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
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
        with _session() as session:
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
                    write_batch(store, var, dataset)
                    cache.release([source])


_MODULE_NAME = 'weatheasy.download'
_LAST = 'last'
_LOG = logging.getLogger(_MODULE_NAME)
_CFS2_DOWNLOADED_FILENAME_TEMPLATE = '{}{}{}.grb2'


def _session() -> Client:
    return Client(transport=HTTPTransport(retries=3), follow_redirects=True)


def _download_cfs2_reanalysis(
    store: Store, end: date, session: Client, download_dir: DownloadCache
):
    download = _Cfs2ReanalysisDownloader(session, download_dir)
    groups = {kind: f'{const.CFS2_REANALYSIS_DIR}/{kind}' for kind in ('flx', 'pgb')}
    for kind in groups:
        set_group_attrs(store, f'{const.CFS2_REANALYSIS_DIR}/{kind}')
    previous = {kind: last_date(store, path) for kind, path in groups.items()}
    date_ = min(
        (value + const.ONE_DAY if value else const.CFS2_REANALYSIS_FIRST_DATE)
        for value in previous.values()
    )
    while date_ < end:
        paths = download(date_)
        if paths:
            for kind, path in groups.items():
                last = previous[kind]
                if last is not None and date_ <= last:
                    continue
                daily = _daily_cfs2(paths, kind, date_)
                write_batch(store, path, daily)
        else:
            _LOG.warning('%s was not found on the server', date_)
        download_dir.release(paths or [])
        date_ += const.ONE_DAY


def _daily_cfs2(paths: list[Path], kind: str, date_: date, template: xr.Dataset | None = None):
    fields = [read_cfs2(path, kind=kind) for path in paths]
    if not fields:
        if template is None:
            msg = 'Cannot determine the grid without any source files'
            raise ValueError(msg)
        daily = xr.full_like(template, np.nan)
    else:
        sample = fields[0]
        # Exact spatial alignment; only the sample time axis may differ.
        combined = xr.concat(fields, dim='time', join='exact')
        bands = const.CFS2_FLX_BANDS if kind == 'flx' else const.CFS2_PGB_BANDS
        data = {}
        for name, info in bands.items():
            values = combined[name].to_numpy()
            if len(paths) < len(const.CFS2_HHS):
                values = np.concatenate(
                    (
                        values,
                        np.full(
                            (len(const.CFS2_HHS) - len(paths), *values.shape[1:]),
                            np.nan,
                            np.float32,
                        ),
                    )
                )
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


def _merge_cfs2_forecast(store: Store, begin: date, end: date, download_dir: Path):
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
                template = read_cfs2(paths[0], kind=kind)
            date_ += const.ONE_DAY
        if template is None:
            msg = f'No forecast files for {kind}; forecast remains unavailable'
            raise ValueError(msg)
        for date_, paths in paths_by_day:
            write_batch(store, group, _daily_cfs2(paths, kind, date_, template))


class _Cfs2ReanalysisDownloader:
    def __init__(self, session: Client, download_dir: DownloadCache) -> None:
        self._session = session
        self._download_dir = download_dir

    def __call__(self, date_: date) -> list[Path] | None:
        ymd = date_.strftime('%Y%m%d')
        ym = ymd[:-2]
        url_template = (
            'https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-analysis/6-hourly-by-pressure/'
            f'{date_.year}/{ym}/{ymd}/cdas1.t{{}}z.pgrbh00.grib2'
        )
        paths = []
        for hhs in const.CFS2_HHS:
            path = self._download_dir.file(f'{ym}/{ymd}.cdas1.t{hhs}z.pgrbh00.grib2')
            if not path.is_file():
                url = url_template.format(hhs)
                _LOG.info('Downloading %s', url)
                response = self._session.get(url, timeout=180)
                if response.status_code == 404:
                    return None
                if not response.is_success:
                    msg = f'Failed to download {url}'
                    raise RuntimeError(msg)
                _LOG.info('Writing %s', path)
                self._download_dir.write(path, response.content)
            self._download_dir.publish([path])
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
