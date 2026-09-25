from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import date
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from requests import Session
from requests.adapters import HTTPAdapter

from weatheasy import const
from weatheasy.era5 import download_era5_data
from weatheasy.sources import read_cfs2, read_netcdf
from weatheasy.storage import last_date, prepare_dataset, write_batch
from weatheasy.util import get_storage, init_parser, utc_now


if TYPE_CHECKING:
    import zarr


def main(*, configure_logging: bool = True) -> None:
    if configure_logging:
        logging.basicConfig(
            format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
            level=logging.INFO,
        )
    parser = init_parser(_MODULE_NAME)
    parser.add_argument(
        '--download-dir',
        type=Path,
        metavar='PATH',
        help='optional local path for persisted downloaded files',
    )
    parser.add_argument(
        'kind',
        choices=('cfs2', 'cmip6', 'era5'),
    )
    parser.add_argument(
        '--area',
        type=float,
        nargs=4,
        metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'),
        help='ERA5-Land geographic area',
    )
    args = parser.parse_args()
    if args.kind == 'era5':
        download_era5_data(
            get_storage(args.data), args.download_dir, area=tuple(args.area) if args.area else None
        )
        return
    if args.area is not None:
        parser.error('--area is only available for era5')
    match args.kind:
        case 'cfs2':
            download = download_cfs2_data
        case 'cmip6':
            download = download_cmip6_data
        case _:
            raise NotImplementedError
    root = get_storage(args.data)
    download(root, args.download_dir)


def download_cfs2_data(root: zarr.Group, download_dir: Path | None = None) -> None:
    group = root.require_group(const.CFS2_DIR)
    today = utc_now().date()

    if updated := group.attrs.get(const.CFS2_KEY_UPDATED):
        time_since_updated = today - date.fromisoformat(str(updated))
        if time_since_updated < const.ONE_DAY:
            _LOG.info('No forecast update required')
            return

    yesterday = today - const.ONE_DAY
    forecast_begin = today - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
    forecast_end = yesterday + const.CFS2_FORECAST_DAYS

    with ExitStack() as stack:
        session = stack.enter_context(_session())
        if download_dir:
            reanalysis_dir = download_dir.joinpath(const.CFS2_REANALYSIS_DIR)
            reanalysis_dir.mkdir(parents=True, exist_ok=True)
            forecast_dir = download_dir.joinpath(const.CFS2_FORECAST_DIR, today.isoformat())
            forecast_dir.mkdir(parents=True, exist_ok=True)
        else:
            reanalysis_dir = stack.enter_context(_temp_dir())
            forecast_dir = stack.enter_context(_temp_dir())
        _download_cfs2_reanalysis(root, forecast_begin, session, reanalysis_dir)
        _download_cfs2_forecast(forecast_begin, yesterday, session, forecast_dir)
        _download_cfs2_forecast(yesterday, forecast_end, session, forecast_dir)
        _merge_cfs2_forecast(root, forecast_begin, forecast_end, forecast_dir)


def download_cmip6_data(root: zarr.Group, download_dir: Path | None = None) -> None:
    _, download_dir = _process_args(const.CMIP6_DIR, root, download_dir)
    with _session() as session, _temp_dir() as temporary:
        for var in const.CMIP6_VARS:
            path = f'{root.path + "/" if root.path else ""}cmip6/{var}'
            previous = last_date(root.store, path)
            first_year = previous.year + 1 if previous else const.CMIP6_FIRST_YEAR
            for year in range(first_year, const.CMIP6_LAST_YEAR + 1):
                filename, content = _load_cmip6_dataset(download_dir, var, year, session)
                source = temporary / filename
                source.write_bytes(content)
                dataset = read_netcdf(source, variables=[var], crs='EPSG:4326')
                expected = np.arange(
                    np.datetime64(f'{year}-01-01'), np.datetime64(f'{year + 1}-01-01')
                )
                if not np.array_equal(dataset.time.to_numpy().astype('datetime64[D]'), expected):
                    msg = f'Incomplete or unexpected CMIP6 dates in {filename}'
                    raise ValueError(msg)
                # Daily CMIP6 timestamps may label the interval at noon. Store
                # calendar days, matching the public date-based query contract.
                dataset = dataset.assign_coords(
                    time=xr.DataArray(
                        expected.astype('datetime64[ns]'), dims='time', attrs=dataset.time.attrs
                    )
                )
                _LOG.info('Saving %s/%s', path, year)
                write_batch(root.store, path, dataset)


_MODULE_NAME = 'weatheasy.download'
_LAST = 'last'
_LOG = logging.getLogger(_MODULE_NAME)
_CFS2_DOWNLOADED_FILENAME_TEMPLATE = '{}{}{}.grb2'


def _process_args(group_name: str, root: zarr.Group, download_dir: Path | None):
    group = root.require_group(group_name)
    if download_dir:
        download_dir /= group_name
        download_dir.mkdir(parents=True, exist_ok=True)
    return group, download_dir


def _session():
    s = Session()
    s.mount('https://', HTTPAdapter(max_retries=3))
    return s


@contextmanager
def _temp_dir():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def _download_cfs2_reanalysis(root: zarr.Group, end: date, session: Session, download_dir: Path):
    download = _Cfs2ReanalysisDownloader(session, download_dir)
    groups = {
        kind: root.require_group(f'{const.CFS2_REANALYSIS_DIR}/{kind}') for kind in ('flx', 'pgb')
    }
    previous = {kind: last_date(root.store, group.path) for kind, group in groups.items()}
    date_ = min(
        (value + const.ONE_DAY if value else const.CFS2_REANALYSIS_FIRST_DATE)
        for value in previous.values()
    )
    while date_ < end:
        paths = download(date_)
        if paths:
            for kind, group in groups.items():
                last = previous[kind]
                if last is not None and date_ <= last:
                    continue
                daily = _daily_cfs2(paths, kind, date_)
                write_batch(root.store, group.path, daily)
        else:
            _LOG.warning('%s was not found on the server', date_)
        date_ += const.ONE_DAY


def _daily_cfs2(paths: list[Path], kind: str, date_: date, template: xr.Dataset | None = None):
    fields = [read_cfs2(path, kind=kind) for path in paths]
    if not fields:
        if template is None:
            msg = 'Cannot determine the grid without any source files'
            raise ValueError(msg)
        daily = xr.full_like(template, np.nan)
        crs = template.spatial_ref.attrs['crs_wkt']
    else:
        sample = fields[0]
        crs = sample.spatial_ref.attrs['crs_wkt']
        # Exact spatial alignment; only the sample time axis may differ.
        for other in fields[1:]:
            for axis in ('latitude', 'longitude'):
                if not np.array_equal(sample[axis], other[axis]):
                    msg = 'CFS2 source grids differ within a day'
                    raise ValueError(msg)
        combined = xr.concat(fields, dim='time', join='exact')
        times = combined.time.to_numpy()
        if len(np.unique(times)) != len(times) or np.any(
            times.astype('datetime64[D]') != np.datetime64(date_)
        ):
            msg = 'CFS2 valid timestamps do not match the requested day'
            raise ValueError(msg)
        bands = const.CFS2_FLX_BANDS if kind == 'flx' else const.CFS2_PGB_BANDS
        _check_pevpr_times(times, date_, bands)
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
    return prepare_dataset(daily, crs=crs)


def _check_pevpr_times(times: np.ndarray, day: date, bands: dict) -> None:
    if 'PEVPR' in bands:
        expected = np.datetime64(day, 'h') + np.array([0, 6, 12, 18], dtype='timedelta64[h]')
        if not np.isin(times, expected).all():
            msg = 'PEVPR requires instantaneous 00/06/12/18 UTC samples'
            raise ValueError(msg)


def _download_cfs2_forecast(
    begin: date,
    end: date,
    session: Session,
    download_dir: Path,
):
    download_dir.mkdir(parents=True, exist_ok=True)
    download = _Cfs2ForecastDownloader(session, download_dir, begin, end)
    download('flx', f'flxf{{}}{{}}.01.{begin:%Y%m%d}00.grb2', const.CFS2_FLX_PARAMS)
    download('pgb', f'pgbf{{}}{{}}.01.{begin:%Y%m%d}00.grb2', const.CFS2_PGB_PARAMS)


def _merge_cfs2_forecast(root: zarr.Group, begin: date, end: date, download_dir: Path):
    # In-place replacement deliberately makes forecast data unavailable. Clear
    # readiness before deleting any chunks, then publish only after both grids.
    parent = root[const.CFS2_DIR]
    parent.attrs.pop(const.CFS2_KEY_UPDATED, None)
    if const.CFS2_FORECAST_DIR in root:
        del root[const.CFS2_FORECAST_DIR]
    for kind in ('flx', 'pgb'):
        group = root.require_group(f'{const.CFS2_FORECAST_DIR}/{kind}')
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
            write_batch(root.store, group.path, _daily_cfs2(paths, kind, date_, template))
    parent.attrs[const.CFS2_KEY_UPDATED] = utc_now().date().isoformat()


class _Cfs2ReanalysisDownloader:
    def __init__(self, session: Session, download_dir: Path) -> None:
        self._session = session
        self._download_dir = download_dir

    def __call__(self, date_: date) -> list[Path] | None:
        ymd = date_.strftime('%Y%m%d')
        ym = ymd[:-2]
        subdir = self._download_dir / ym
        subdir.mkdir(exist_ok=True)
        url_template = (
            'https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-analysis/6-hourly-by-pressure/'
            f'{date_.year}/{ym}/{ymd}/cdas1.t{{}}z.pgrbh00.grib2'
        )
        paths = []
        for hhs in const.CFS2_HHS:
            path = subdir / f'{ymd}.cdas1.t{hhs}z.pgrbh00.grib2'
            if not path.is_file():
                url = url_template.format(hhs)
                _LOG.info('Downloading %s', url)
                with self._session.get(url, timeout=180) as response:
                    if response.status_code == 404:
                        return None
                    if not response.ok:
                        msg = f'Failed to download {url}'
                        raise RuntimeError(msg)
                    _LOG.info('Writing %s', path)
                    path.write_bytes(response.content)
            paths.append(path)
        return paths


class _Cfs2ForecastDownloader:
    _base_url: ClassVar = 'https://nomads.ncep.noaa.gov/cgi-bin/'
    _min_interval: ClassVar = 1.0 / 3.0

    def __init__(self, session: Session, download_dir: Path, begin: date, end: date) -> None:
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
            path = self._download_dir / _CFS2_DOWNLOADED_FILENAME_TEMPLATE.format(
                kind, date_str, hhs
            )
            if path.is_file():
                _LOG.info('Skipping already downloaded %s', path)
                continue
            elapsed = time.time() - self._last_call
            delta = self._min_interval - elapsed
            if delta > 0:
                time.sleep(delta)
            self._last_call = time.time()
            _LOG.info('Downloading %s', file)
            params['file'] = file
            with self._session.get(url, params=params, timeout=180) as response:
                if response.ok:
                    if response.content.startswith(b'<!doctype html>'):
                        _LOG.critical('Exceeded overrate limit for nomads.ncep.noaa.gov')
                        sys.exit(1)
                    _LOG.info('Writing %s', path)
                    path.write_bytes(response.content)
                else:
                    _LOG.critical('Failed to download %s', response.url)


def _cfs2_forecast_dates(date_: date, end: date):
    while date_ < end:
        yield date_.strftime('%Y%m%d')
        date_ += const.ONE_DAY


def _load_cmip6_dataset(path: Path | None, var: str, year: int, session: Session):
    filename = f'{var}_{year}.nc'
    if path:
        path /= filename

    if path and path.is_file():
        _LOG.info('Reading %s', path)
        return filename, path.read_bytes()

    kind = 'historical' if year <= const.CMIP6_LAST_HISTORICAL_YEAR else 'ssp245'
    url = (
        'https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/ACCESS-CM2/'
        f'{kind}/r1i1p1f1/{var}/{var}_day_ACCESS-CM2_{kind}_r1i1p1f1_gn_{year}.nc'
    )
    _LOG.info('Downloading %s', url)
    with session.get(url, timeout=180) as response:
        if not response.ok:
            _LOG.critical('Failed to download %s', url)
            sys.exit(1)
        if path:
            _LOG.info('Writing %s', path)
            path.write_bytes(response.content)

    return filename, response.content


if __name__ == '__main__':
    main()
