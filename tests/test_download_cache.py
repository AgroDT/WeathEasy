from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem

from weatheasy import const, download
from weatheasy.file_storage import FileStore
from weatheasy.raw_cache import download_cache
from weatheasy.storage import get_store


class TrackingFileSystem(MemoryFileSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reads = []

    def open(self, path: str, mode: str = 'rb', **kwargs):
        if mode == 'rb':
            self.reads.append(path)
        return super().open(path, mode, **kwargs)


def test_remote_cache_publishes_download_and_fetches_once(tmp_path: Path):
    fs = TrackingFileSystem(skip_instance_cache=True)
    remote = FileStore(fs, f'{tmp_path.name}/raw')
    with download_cache(remote, 'era5') as cache:
        path = cache.file('month.nc')
        path.write_bytes(b'complete monthly data')
        assert not remote.child('era5').exists('month.nc')
        cache.publish([path])
        assert cache.file('month.nc') == path
        assert path.read_bytes() == b'complete monthly data'
        assert fs.reads == []
    assert not path.exists()
    with download_cache(remote, 'era5') as cache:
        path = cache.file('month.nc')
        assert path.read_bytes() == b'complete monthly data'
        assert cache.file('month.nc') == path
        cache.publish([path])
    assert fs.reads == [f'{remote.root}/era5/month.nc']


def test_local_cache_uses_original_path_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def unexpected_copy(*_args, **_kwargs):
        pytest.fail('local cache must not copy files')

    monkeypatch.setattr('weatheasy.raw_cache.shutil.copyfileobj', unexpected_copy)
    original = tmp_path / 'cmip6' / 'tas.nc'
    original.parent.mkdir()
    original.write_bytes(b'original')
    with download_cache(tmp_path, 'cmip6') as cache:
        assert cache.file('tas.nc') == original
        cache.publish([original])
    assert original.read_bytes() == b'original'


def test_incomplete_download_does_not_publish(tmp_path: Path):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')

    def interrupted_download():
        with download_cache(remote) as cache:
            cache.file('bad.nc').write_bytes(b'incomplete')
            msg = 'download interrupted'
            raise ValueError(msg)

    with pytest.raises(ValueError, match='download interrupted'):
        interrupted_download()
    assert not remote.exists('bad.nc')


@pytest.mark.parametrize('remote', [False, True])
def test_cmip6_public_download_processes_working_file_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, remote: bool
):
    times = np.arange('2000-01-01', '2001-01-01', dtype='datetime64[D]')
    source = tmp_path / 'fixture.nc'
    xr.Dataset(
        {'tas': (('time', 'lat', 'lon'), np.full((366, 2, 2), 280.0), {'units': 'K'})},
        coords={'time': times, 'lat': [0.125, 0.375], 'lon': [0.125, 0.375]},
    ).to_netcdf(source)
    content = source.read_bytes()
    fs = TrackingFileSystem(skip_instance_cache=True)
    backing = FileStore(fs, f'{tmp_path.name}/downloads')
    location = backing if remote else tmp_path / 'downloads'
    monkeypatch.setattr(const, 'CMIP6_VARS', ['tas'])
    monkeypatch.setattr(const, 'CMIP6_FIRST_YEAR', 2000)
    monkeypatch.setattr(const, 'CMIP6_LAST_YEAR', 2000)
    requests = []

    def get(url: str, **_kwargs):
        requests.append(url)
        return SimpleNamespace(is_success=True, content=content)

    # A real context manager with only the external HTTP boundary replaced.
    monkeypatch.setattr(download, '_session', lambda: nullcontext(SimpleNamespace(get=get)))
    original_read = download.read_netcdf
    processed = []

    def read(path: Path, **kwargs):
        processed.append(path)
        if remote and len(processed) == 1:
            assert backing.child('cmip6').exists('tas_2000.nc')
            assert fs.reads == []
        if not remote:
            assert path == tmp_path / 'downloads' / 'cmip6' / 'tas_2000.nc'
        if len(processed) == 1:
            msg = 'processing interrupted'
            raise ValueError(msg)
        return original_read(path, **kwargs)

    monkeypatch.setattr(download, 'read_netcdf', read)
    with pytest.raises(ValueError, match='processing interrupted'):
        download.download_cmip6_data(get_store(tmp_path / 'prepared1'), location)
    download.download_cmip6_data(get_store(tmp_path / 'prepared1'), location)
    if remote:
        assert backing.child('cmip6').exists('tas_2000.nc')
        assert fs.reads == [f'{backing.root}/cmip6/tas_2000.nc']
    download.download_cmip6_data(get_store(tmp_path / 'prepared2'), location)
    assert len(requests) == 1
    assert len(processed) == 3
    if remote:
        assert fs.reads == [f'{backing.root}/cmip6/tas_2000.nc'] * 2


def test_cfs2_downloaders_reuse_remote_files_without_http(tmp_path: Path):
    fs = TrackingFileSystem(skip_instance_cache=True)
    remote = FileStore(fs, f'{tmp_path.name}/raw')
    calls = []

    def get(url: str, **_kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200, is_success=True, content=b'GRIB')

    session = SimpleNamespace(get=get)
    for attempt in range(2):
        with download_cache(remote, 'reanalysis') as cache:
            paths = download._Cfs2ReanalysisDownloader(session, cache)(date(2000, 1, 1))
            assert len(paths) == 4
            assert all(path.read_bytes() == b'GRIB' for path in paths)
            assert len(remote.child('reanalysis').glob('**/*.grib2')) == 4
        with download_cache(remote, 'forecast') as cache:
            downloader = download._Cfs2ForecastDownloader(
                session, cache, date(2000, 1, 1), date(2000, 1, 2)
            )
            downloader._min_interval = 0
            downloader('flx', 'file{}{}', {})
            paths = list(cache.root.glob('*.grb2'))
            assert len(paths) == 4
            assert all(path.read_bytes() == b'GRIB' for path in paths)
            assert len(remote.child('forecast').glob('*.grb2')) == 4
        assert len(calls) == 8
        assert len(fs.reads) == attempt * 8


def test_upload_failure_leaves_cmip6_preparation_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from weatheasy.storage import last_date

    monkeypatch.setattr(const, 'CMIP6_VARS', ['tas'])
    monkeypatch.setattr(const, 'CMIP6_FIRST_YEAR', 2000)
    monkeypatch.setattr(const, 'CMIP6_LAST_YEAR', 2000)
    monkeypatch.setattr(
        download,
        '_session',
        lambda: nullcontext(
            SimpleNamespace(
                get=lambda *_args, **_kwargs: SimpleNamespace(is_success=True, content=b'input')
            )
        ),
    )

    def unexpected_read(*_args, **_kwargs):
        pytest.fail('processing must start only after cache publication')

    monkeypatch.setattr(download, 'read_netcdf', unexpected_read)

    def fail_upload(*_args):
        msg = 'upload failed'
        raise OSError(msg)

    monkeypatch.setattr('weatheasy.raw_cache.DownloadCache.publish', fail_upload)
    store = get_store(tmp_path / 'prepared')
    with pytest.raises(OSError, match='upload failed'):
        download.download_cmip6_data(store, tmp_path / 'downloads')
    assert last_date(store, 'tas') is None
