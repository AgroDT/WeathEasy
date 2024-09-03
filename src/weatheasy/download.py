from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TYPE_CHECKING, ClassVar

import netCDF4 as nc
import numpy as np
import rasterio
import zarr
from requests import Session
from requests.adapters import HTTPAdapter

from weatheasy import const
from weatheasy.util import get_storage, init_parser, utc_now


if TYPE_CHECKING:
    from rasterio.coords import BoundingBox


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
        choices=('cfs2', 'cmip6'),
    )
    args = parser.parse_args()
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
        time_since_updated = today - date.fromisoformat(updated)
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

    group.attrs[const.CFS2_KEY_UPDATED] = today.isoformat()


def download_cmip6_data(root: zarr.Group, download_dir: Path | None = None) -> None:
    years_key = 'years'

    height, width = _get_size(const.CMIP6_RESOLUTION, const.CMIP6_BBOX)
    height += 1
    width += 1

    total_days = (date(const.CMIP6_LAST_YEAR, 12, 31) - date(const.CMIP6_FIRST_YEAR, 1, 1)).days
    arr_shape = total_days, height, width
    arr_chunks = _FOUR_YEAR_DAYS, 100, 100

    group, download_dir = _process_args(const.CMIP6_DIR, root, download_dir)
    buffer = np.full((_FOUR_YEAR_DAYS, height, width), np.nan, np.float32)

    with _session() as session:
        for var in const.CMIP6_VARS:
            array = group.require_dataset(
                name=var,
                shape=arr_shape,
                dtype=np.float32,
                chunks=arr_chunks,
                fill_value=np.nan,
            )
            first_year: int = array.attrs.get(years_key, (None, const.CMIP6_FIRST_YEAR - 1))[1] + 1
            total_day_offset = (date(first_year, 1, 1) - date(const.CMIP6_FIRST_YEAR, 1, 1)).days
            for year in range(first_year, const.CMIP6_LAST_YEAR + 1, 4):
                buffer[:] = 0
                day_offset = 0
                for next_year in range(year, min(year + 4, const.CMIP6_LAST_YEAR + 1)):
                    filename, nc_bytes = _load_cmip6_dataset(download_dir, var, next_year, session)
                    with nc.Dataset(filename, memory=nc_bytes, filling=False) as ds:
                        resolution = float(ds.resolution_id.split(' ', 1)[0])
                        lon = ds.variables['lon']
                        lat = ds.variables['lat']
                        bbox = float(lon[0]), float(lat[0]), float(lon[-1]), float(lat[-1])
                        if resolution != const.CMIP6_RESOLUTION or bbox != const.CMIP6_BBOX:
                            _LOG.critical('Unexpected shape or geo referencing %s', filename)
                            sys.exit(1)
                        data = ds.variables[var]
                        day_count = data.shape[0]
                        buffer[day_offset:day_offset + day_count] = (
                            data[:].filled(fill_value=np.nan))  # fmt: skip
                        day_offset += day_count
                _LOG.info('Saving %s[%d:%d]', array.path, year, next_year)
                if day_offset != _FOUR_YEAR_DAYS:
                    day_offset -= 1
                array[total_day_offset : total_day_offset + _FOUR_YEAR_DAYS] = buffer[:day_offset]
                array.attrs[years_key] = const.CMIP6_FIRST_YEAR, next_year
                total_day_offset += _FOUR_YEAR_DAYS


_MODULE_NAME = __package__ + '.download'
_FOUR_YEAR_DAYS = 1461
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


def _download_cfs2_reanalysis(  # noqa: PLR0915
    root: zarr.Group,
    end: date,
    session: Session,
    download_dir: Path,
):
    group = root.require_group(const.CFS2_REANALYSIS_DIR)
    if last := group.attrs.get(_LAST):
        date_ = date.fromisoformat(last) + const.ONE_DAY
    else:
        date_ = const.CFS2_REANALYSIS_FIRST_DATE

    height, width = _get_size(const.CFS2_REANALYSIS_RESOLUTION[0], const.CFS2_REANALYSIS_BBOX)

    day_dimension = (
        end.replace(month=12, day=31)
        - const.CFS2_REANALYSIS_FIRST_DATE.replace(month=1, day=1)
    ).days + 1  # fmt: skip
    array_shape = day_dimension, height, width
    arrays = list(_get_cfs2_arrays(group, array_shape, (_FOUR_YEAR_DAYS, 100, 100)))

    tmp_group = group.require_group('_tmp')
    tmp_arrays = list(_get_cfs2_arrays(tmp_group, array_shape, (1, height, width)))

    download = _Cfs2ReanalysisDownloader(session, download_dir)

    first_day = (date_ - const.CFS2_REANALYSIS_FIRST_DATE).days
    total_days = (end - const.CFS2_REANALYSIS_FIRST_DATE).days

    day0 = first_day % _FOUR_YEAR_DAYS
    # day0_ is used to continue interrupted downloads
    day0_ = tmp_group.attrs.get(_LAST)
    if isinstance(day0_, int):
        day0_ += 1
        date_ += timedelta(days=day0_ - day0)
    else:
        day0_ = day0

    day1 = min(_FOUR_YEAR_DAYS, day0_ + total_days - first_day)
    day_buffer_shape = len(const.CFS2_HHS), height, width
    last_success: date | None = None

    while date_ < end:
        day_uploader = _Cfs2ReanalysisDayUploader(day_buffer_shape, tmp_group, tmp_arrays)
        day_uploader.start()
        for day in range(day0_, day1):
            try:
                paths = download(date_)
            except Exception as err:
                _LOG.critical(err)
                day_uploader.join()
                sys.exit(1)
            if paths:
                last_success = date_
                day_uploader.enqueue(day, paths)
            else:
                _LOG.warning('%s was not found on the server', date_)
            date_ += const.ONE_DAY
        day_uploader.join()
        last_day = first_day + day1 - day0
        if last_success is None:
            _LOG.warning('Failed to download any reanalysis data')
            return
        for (_, array), (_, tmp_array) in zip(arrays, tmp_arrays, strict=True):
            _LOG.info('Saving %s[%d:%d]', array.path, first_day, last_day)
            array[first_day:last_day] = tmp_array[day0:day1]
            tmp_array[:] = 0.0
        first_day = last_day
        day0 = 0
        day0_ = 0
        day1 = min(_FOUR_YEAR_DAYS, total_days - first_day)
        tmp_group.attrs.pop(_LAST, None)
        group.attrs[_LAST] = last_success.isoformat()

    tmp_group.clear()


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


def _merge_cfs2_forecast(
    root: zarr.Group,
    begin: date,
    end: date,
    download_dir: Path,
):
    group = root.require_group(const.CFS2_FORECAST_DIR)
    merge = _Cfs2ForecastMerger(group, download_dir, begin, end)
    merge('flx', const.CFS2_FLX_BANDS, const.CFS2_FLX_RESOLUTION, const.CFS2_FLX_BBOX)
    merge('pgb', const.CFS2_PGB_BANDS, const.CFS2_PGB_RESOLUTION, const.CFS2_PGB_BBOX)


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


class _Cfs2ReanalysisDayUploader(Thread):
    def __init__(
        self,
        buffer_shape: tuple[int, int, int],
        group: zarr.Group,
        arrays: list[tuple[str, zarr.Array]],
    ) -> None:
        super().__init__()
        self._queue: Queue[tuple[int, list[Path]] | None] = Queue()
        self._buffer = np.full(buffer_shape, np.nan, np.float32)
        self._group = group
        self._arrays = arrays

    def enqueue(self, day: int, paths: list[Path]) -> None:
        self._queue.put((day, paths))

    def join(self, timeout: float | None = None) -> None:
        self._queue.put(None)
        super().join(timeout)

    def run(self) -> None:
        while task := self._queue.get():
            self._upload_day(*task)

    def _upload_day(self, day: int, paths: list[Path]):
        buffer = self._buffer
        with ExitStack() as stack:
            dss: list[rasterio.DatasetReader] = []
            for path in paths:
                ds = stack.enter_context(rasterio.open(path, sharing=True))
                if (
                    ds.res != const.CFS2_REANALYSIS_RESOLUTION
                    or ds.bounds != const.CFS2_REANALYSIS_BBOX
                ):
                    err = f'Unexpected shape or geo referencing {path}'
                    raise ValueError(err)
                dss.append(ds)
            for var, array in self._arrays:
                buffer[:] = np.nan
                var_bands = const.CFS2_BANDS[var]
                for i, ds in enumerate(dss):
                    ds.read(var_bands.reanalysis, out=buffer[i])
                _LOG.info('Saving %s[%d]', array.path, day)
                array[day] = var_bands.daily_stat(buffer, axis=0)
        self._group.attrs[_LAST] = day


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


class _Cfs2ForecastMerger:
    def __init__(
        self,
        group: zarr.Group,
        download_dir: Path,
        begin: date,
        end: date,
    ) -> None:
        self._group = group
        self._download_dir = download_dir
        self._begin = begin
        self._end = end

    def __call__(
        self,
        kind: str,
        bands: dict[str, const.Cfs2Band],
        resolution: tuple[float, float],
        bbox: BoundingBox,
    ) -> None:
        with ExitStack() as stack:
            day_dss = []
            for date_str in _cfs2_forecast_dates(self._begin, self._end):
                hhs_dss = []
                for hhs in const.CFS2_HHS:
                    path = self._download_dir / _CFS2_DOWNLOADED_FILENAME_TEMPLATE.format(
                        kind, date_str, hhs
                    )
                    if path.is_file():
                        ds = stack.enter_context(rasterio.open(path))
                        if ds.res != resolution or ds.bounds != bbox:
                            _LOG.critical('Unexpected shape or geo referencing %s', path)
                            sys.exit(1)
                        hhs_dss.append(ds)
                day_dss.append(hhs_dss)

            height, width = _get_size(resolution[0], bbox)
            day_buffer_shape = len(const.CFS2_HHS), height, width
            day_buffer = np.full(day_buffer_shape, np.nan, np.float32)
            var_buffer_shape = (self._end - self._begin).days, height, width
            var_buffer = np.full(var_buffer_shape, np.nan, np.float32)

            for var, var_bands in bands.items():
                var_buffer[:] = np.nan
                for day, hhs_dss in enumerate(day_dss):
                    day_buffer[:] = np.nan
                    for i, ds in enumerate(hhs_dss):
                        ds.read(var_bands.forecast, out=day_buffer[i])
                    var_buffer[day] = var_bands.daily_stat(day_buffer, axis=0)
                array = self._group.array(
                    name=var,
                    data=var_buffer,
                    chunks=(None, 100, 100),
                    overwrite=True,
                    fill_value=np.nan,
                )
                _LOG.info('Saved %s', array.path)


def _get_cfs2_arrays(group: zarr.Group, shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for var in const.CFS2_BANDS:
        array = group.get(var)
        if isinstance(array, zarr.Array):
            if array.shape != shape:
                array.resize(*shape)
        else:
            array = group.require_dataset(
                name=var,
                shape=shape,
                dtype=np.float32,
                chunks=chunks,
                fill_value=np.nan,
            )
        yield var, array


def _cfs2_forecast_dates(date_: date, end: date):
    while date_ < end:
        yield date_.strftime('%Y%m%d')
        date_ += const.ONE_DAY


def _get_size(resolution: float, bbox: BoundingBox) -> tuple[int, int]:
    return (
        math.ceil((bbox.top - bbox.bottom) / resolution),
        math.ceil((bbox.right - bbox.left) / resolution),
    )


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
