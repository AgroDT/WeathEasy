import json
import os
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem

from weatheasy import const, download, raw_cache
from weatheasy.file_storage import FileStore
from weatheasy.raw_cache import download_cache
from weatheasy.storage import get_store, open_dataset, prepare_dataset, set_group_attrs, write_batch


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local-app-data'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('USERPROFILE', str(tmp_path / 'home'))


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
        cache.release([path])
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
@pytest.mark.parametrize('full_year', [False, True])
def test_cmip6_public_download_processes_working_file_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, remote: bool, full_year: bool
):
    days = (
        np.arange('2000-01-01', '2001-01-01', dtype='datetime64[D]')
        if full_year
        else np.array(['2000-02-28', '2000-02-29', '2000-03-02'], dtype='datetime64[D]')
    )
    times = days.astype('datetime64[h]') + np.timedelta64(12, 'h')
    source = tmp_path / 'fixture.nc'
    xr.Dataset(
        {'tas': (('time', 'lat', 'lon'), np.full((len(times), 2, 2), 280.0), {'units': 'K'})},
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
            old = time.time() - 30 * 86400
            os.utime(path, (old, old))
            obsolete = path.parent / 'tas_1999.nc'
            obsolete.write_bytes(b'obsolete')
            os.utime(obsolete, (old, old))
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
        assert fs.reads == []
        assert not processed[0].exists()
        assert not (processed[0].parent / 'tas_1999.nc').exists()
    download.download_cmip6_data(get_store(tmp_path / 'prepared2'), location)
    with open_dataset(get_store(tmp_path / 'prepared2'), 'tas') as result:
        np.testing.assert_array_equal(result.time, days.astype('datetime64[ns]'))
        np.testing.assert_allclose(result.tas, 280)
    assert len(requests) == 1
    assert len(processed) == 3
    if remote:
        assert fs.reads == [f'{backing.root}/cmip6/tas_2000.nc']


def test_cfs2_downloaders_reuse_remote_files_without_http(tmp_path: Path):
    fs = TrackingFileSystem(skip_instance_cache=True)
    remote = FileStore(fs, f'{tmp_path.name}/raw')
    calls = []

    def get(url: str, **_kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200, is_success=True, content=b'GRIB')

    session = SimpleNamespace(get=get)
    for _attempt in range(2):
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
        assert fs.reads == []


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


def test_cleanup_keeps_old_needed_files_and_local_cache(tmp_path: Path):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    with download_cache(remote, 'era5', work_dir=tmp_path / 'work') as cache:
        needed = cache.file('needed.grib')
        stale = cache.file('obsolete.grib')
        recent = cache.file('recent.grib')
        partial = cache.root / 'needed.grib.part'
        for path in (needed, stale, recent, partial):
            path.write_bytes(b'data')
        old = time.time() - 30 * 86400
        for path in (needed, stale):
            os.utime(path, (old, old))
        metadata = cache.root / '.weatheasy-cache.json'
        os.utime(metadata, (old, old))
        cache.cleanup(lambda name: name == 'needed.grib')
        assert metadata.is_file()
        assert needed.read_bytes() == b'data'
        assert recent.read_bytes() == b'data'
        assert not stale.exists()
        assert not partial.exists()
        cache.release([needed])
        assert not needed.exists()

    with download_cache(tmp_path / 'local', 'era5', work_dir=tmp_path / 'unused') as cache:
        path = cache.file('obsolete.grib')
        path.write_bytes(b'original')
        os.utime(path, (old, old))
        cache.cleanup(lambda _name: False, max_age_days=0)
        cache.release([path])
    assert path.read_bytes() == b'original'
    assert not (tmp_path / 'unused').exists()


def test_upload_retry_uses_retained_download_without_remote_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fs = TrackingFileSystem(skip_instance_cache=True)
    remote = FileStore(fs, f'{tmp_path.name}/raw')
    publish = FileStore.write_atomic

    @contextmanager
    def fail_upload(_self, _name):  # noqa: ANN001
        msg = 'upload interrupted'
        raise OSError(msg)
        yield  # Make failure occur when entering the upload context.

    monkeypatch.setattr(FileStore, 'write_atomic', fail_upload)
    with download_cache(remote, 'cmip6', work_dir=tmp_path / 'work') as cache:
        path = cache.file('tas_2000.nc')
        cache.write(path, b'complete provider response')
        with pytest.raises(OSError, match='upload interrupted'):
            cache.publish([path])
    assert path.read_bytes() == b'complete provider response'
    assert not remote.child('cmip6').exists('tas_2000.nc')
    monkeypatch.setattr(FileStore, 'write_atomic', publish)
    with download_cache(remote, 'cmip6', work_dir=tmp_path / 'work') as cache:
        assert cache.file('tas_2000.nc') == path
        cache.publish([path])
    assert fs.reads == []
    with remote.child('cmip6').open('tas_2000.nc') as stream:
        assert stream.read() == b'complete provider response'


def test_working_files_are_separated_by_remote_and_result_store(tmp_path: Path):
    fs = MemoryFileSystem()
    roots = []
    for remote, target in (
        ('one/raw', 'first.zarr'),
        ('two/raw', 'first.zarr'),
        ('one/raw', 'second.zarr'),
    ):
        with download_cache(
            FileStore(fs, remote), 'era5', target=target, work_dir=tmp_path
        ) as cache:
            roots.append(cache.root)
            settings = json.loads((cache.root / '.weatheasy-cache.json').read_text())
            assert settings['download_cache'] == remote
            assert settings['target'] == target
            cache.file('same.grib').write_bytes(target.encode())
    assert len(set(roots)) == 3
    with download_cache(
        FileStore(fs, 'one/raw'), 'era5', target='first.zarr', work_dir=tmp_path
    ) as cache:
        assert cache.file('same.grib').read_bytes() == b'first.zarr'


def test_provider_write_failure_never_exposes_partial_as_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    original_write = Path.write_bytes

    def interrupt(path, content):  # noqa: ANN001
        original_write(path, content[:3])
        msg = 'disk write interrupted'
        raise OSError(msg)

    with download_cache(remote, 'cmip6', work_dir=tmp_path / 'work') as cache:
        path = cache.file('tas.nc')
        monkeypatch.setattr(Path, 'write_bytes', interrupt)
        with pytest.raises(OSError, match='disk write interrupted'):
            cache.write(path, b'complete response')
        assert not cache.exists('tas.nc')
        assert not list(cache.root.glob('*.part'))
    monkeypatch.setattr(Path, 'write_bytes', original_write)
    with download_cache(remote, 'cmip6', work_dir=tmp_path / 'work') as cache:
        path = cache.file('tas.nc')
        cache.write(path, b'complete response')
        assert path.read_bytes() == b'complete response'


@pytest.mark.parametrize('already_updated', [False, True])
def test_cfs2_cleanup_respects_both_grids_and_current_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    already_updated: bool,
):
    monkeypatch.setattr(download, 'utc_now', lambda: datetime(2000, 1, 10, tzinfo=UTC))
    store = get_store(tmp_path / 'prepared')
    for kind, day in (('flx', '2000-01-02'), ('pgb', '2000-01-01')):
        daily = xr.Dataset(
            {
                'temperature': (
                    ('time', 'latitude', 'longitude'),
                    np.ones((1, 2, 2)),
                    {'units': 'K'},
                )
            },
            coords={
                'time': [np.datetime64(day)],
                'latitude': [0.125, 0.375],
                'longitude': [0.125, 0.375],
            },
        )
        write_batch(store, f'reanalysis/{kind}', prepare_dataset(daily, crs='EPSG:4326'))
    if already_updated:
        set_group_attrs(store, '', updated='2000-01-10')
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    old = time.time() - 30 * 86400
    with download_cache(
        remote, 'reanalysis', work_dir=tmp_path / 'work', target=str(store)
    ) as cache:
        done = cache.file('200001/20000101.cdas1.t00z.pgrbh00.grib2')
        needed = cache.file('200001/20000102.cdas1.t00z.pgrbh00.grib2')
    with download_cache(remote, 'forecast', work_dir=tmp_path / 'work', target=str(store)) as cache:
        forecast = cache.file('2000-01-10/flx2000010700.grb2')
        obsolete = cache.file('2000-01-09/flx2000010700.grb2')
    for path in (done, needed, forecast, obsolete):
        path.write_bytes(b'GRIB')
        os.utime(path, (old, old))

    def stop_before_network():
        msg = 'downloads not started'
        raise OSError(msg)

    monkeypatch.setattr(download, '_session', stop_before_network)
    if already_updated:
        download.download_cfs2_data(store, remote, work_dir=tmp_path / 'work')
    else:
        with pytest.raises(OSError, match='downloads not started'):
            download.download_cfs2_data(store, remote, work_dir=tmp_path / 'work')
    assert not done.exists()
    assert needed.exists() is not already_updated
    assert forecast.exists() is not already_updated
    assert not obsolete.exists()
    assert not obsolete.parent.exists()


@pytest.mark.parametrize(
    ('platform', 'environment', 'relative'),
    [
        ('win32', {'LOCALAPPDATA': 'local', 'XDG_STATE_HOME': 'xdg'}, 'local'),
        ('win32', {'XDG_STATE_HOME': 'xdg'}, 'home/AppData/Local'),
        ('win32', {'LOCALAPPDATA': ''}, 'home/AppData/Local'),
        (
            'darwin',
            {'LOCALAPPDATA': 'local', 'XDG_STATE_HOME': 'xdg'},
            'home/Library/Application Support',
        ),
        ('linux', {'XDG_STATE_HOME': 'xdg', 'LOCALAPPDATA': 'local'}, 'xdg'),
        ('linux', {}, 'home/.local/state'),
        ('linux', {'XDG_STATE_HOME': ''}, 'home/.local/state'),
    ],
)
def test_default_work_dir_uses_platform_conventions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    environment: dict[str, str],
    relative: str,
):
    monkeypatch.setattr(raw_cache.sys, 'platform', platform)
    monkeypatch.setattr(Path, 'home', classmethod(lambda _cls: tmp_path / 'home'))
    for key in ('XDG_STATE_HOME', 'LOCALAPPDATA'):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, str(tmp_path / value) if value else '')
    assert raw_cache.default_work_dir() == tmp_path / relative / 'weatheasy' / 'downloads'


@pytest.mark.parametrize('platform', ['linux', 'win32', 'darwin'])
def test_explicit_work_dir_bypasses_platform_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
):
    monkeypatch.setattr(raw_cache.sys, 'platform', platform)

    def unexpected_default():
        pytest.fail('explicit work directory must bypass platform defaults')

    monkeypatch.setattr(raw_cache, 'default_work_dir', unexpected_default)
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    explicit = tmp_path / 'explicit'
    with download_cache(remote, 'era5', work_dir=explicit) as cache:
        path = cache.file('month.grib')
        cache.write(path, b'GRIB')
        assert path.is_relative_to(explicit)
        assert path.read_bytes() == b'GRIB'
