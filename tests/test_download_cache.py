import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem
from httpx2 import (
    Client,
    MockTransport,
    ReadError,
    ReadTimeout,
    RemoteProtocolError,
    Request,
    Response,
    SyncByteStream,
)

from weatheasy import const, download, raw_cache
from weatheasy.file_storage import FileStore
from weatheasy.raw_cache import download_cache
from weatheasy.storage import get_store, open_dataset, prepare_dataset, set_group_attrs, write_batch


@pytest.fixture(name='isolated_work_dir', autouse=True)
def init_isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local-app-data'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('USERPROFILE', str(tmp_path / 'home'))


@pytest.fixture(name='grib_content')
def get_grib_content() -> bytes:
    return Path(__file__).with_name('fixtures').joinpath('era5', 'day.grib').read_bytes()


@pytest.fixture(name='working_cache_files')
def create_working_cache_files(tmp_path: Path):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    with download_cache(remote, 'era5', work_dir=tmp_path / 'work') as cache:
        paths = {name: cache.file(f'{name}.grib') for name in ('needed', 'obsolete', 'recent')}
        paths['partial'] = cache.root / 'needed.grib.part'
        paths['metadata'] = cache.root / '.weatheasy-cache.json'
        for name in ('needed', 'obsolete', 'recent', 'partial'):
            paths[name].write_bytes(b'data')
        old = time.time() - 30 * 86400
        for name in ('needed', 'obsolete', 'metadata'):
            os.utime(paths[name], (old, old))
        yield cache, paths


class TrackingFileSystem(MemoryFileSystem):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reads = []

    def open(
        self,
        path: str,
        mode: str = 'rb',
        block_size: int | None = None,
        cache_options: dict | None = None,
        compression: str | None = None,
        **kwargs,
    ):
        if mode == 'rb':
            self.reads.append(path)
        return super().open(path, mode, block_size, cache_options, compression, **kwargs)


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


@pytest.mark.parametrize('remote', [False, True], ids=['local-cache', 'remote-cache'])
@pytest.mark.parametrize('full_year', [False, True], ids=['partial-year', 'full-year'])
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
    monkeypatch.setattr(download, '_session', lambda _proxy: nullcontext(SimpleNamespace(get=get)))
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


@pytest.mark.parametrize('error_type', [RemoteProtocolError, ReadError, ReadTimeout])
@pytest.mark.parametrize('failures', [2, 4], ids=['recovers', 'exhausted'])
def test_cfs2_reanalysis_retries_interrupted_body_without_caching_partial_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    failures: int,
    grib_content: bytes,
):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    calls = []
    delays = []
    monkeypatch.setattr(download.time, 'sleep', delays.append)

    class InterruptedBody(SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'GRIB-partial'
            msg = 'incomplete response body'
            raise error_type(msg)

    def respond(request: Request) -> Response:
        calls.append(request.url.path)
        if len(calls) <= failures:
            return Response(200, stream=InterruptedBody())
        return Response(200, content=grib_content)

    with (
        download_cache(remote, work_dir=tmp_path / 'work') as cache,
        Client(transport=MockTransport(respond)) as session,
    ):
        downloader = download._Cfs2ReanalysisDownloader(session, cache)
        if failures == 4:
            with pytest.raises(error_type, match='incomplete response body'):
                downloader(date(2018, 10, 15))
            assert len(calls) == 4
            assert remote.glob('**/*.grib2') == []
            assert list(cache.root.rglob('*.grib2')) == []
            assert list(cache.root.rglob('*.part')) == []
        paths = downloader(date(2018, 10, 15))
        assert len(paths) == 4
        assert all(path.read_bytes() == grib_content for path in paths)
        assert len(remote.glob('**/*.grib2')) == 4
        assert len(calls) == failures + 4
        assert delays == ([1, 2] if failures == 2 else [1, 2, 4])


@pytest.mark.parametrize('cached', [None, 'local', 'remote'])
@pytest.mark.parametrize('empty_response', [False, True])
def test_cfs2_reanalysis_retries_empty_cache_and_skips_empty_responses(
    tmp_path: Path, cached: str | None, grib_content: bytes, *, empty_response: bool
):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    name = '201810/20181015.cdas1.t06z.pgrbh00.grib2'
    calls = []

    def respond(request: Request) -> Response:
        calls.append(request.url.path)
        if empty_response and '.t06z.' in request.url.path:
            return Response(200, content=b'')
        return Response(200, content=grib_content)

    with (
        download_cache(remote, work_dir=tmp_path / 'work') as cache,
        Client(transport=MockTransport(respond)) as session,
    ):
        if cached == 'local':
            cache.file(name).write_bytes(b'')
        elif cached == 'remote':
            with remote.write_atomic(name) as stream:
                stream.write(b'')
        downloader = download._Cfs2ReanalysisDownloader(session, cache)
        paths = downloader(date(2018, 10, 15))
        assert len(calls) == 4
        assert len(paths) == (3 if empty_response else 4)
        assert all(path.read_bytes() == grib_content for path in paths)
        if empty_response:
            assert not (cache.root / name).exists()
            if remote.exists(name):
                with remote.open(name) as stream:
                    assert stream.read() == b''
            assert len(remote.glob('**/*.grib2')) == (4 if cached == 'remote' else 3)
        else:
            with remote.open(name) as stream:
                assert stream.read() == grib_content
            assert downloader(date(2018, 10, 15)) == paths
            assert len(calls) == 4


@pytest.mark.parametrize('cached', ['local', 'remote'])
def test_cfs2_reanalysis_replaces_truncated_cache(tmp_path: Path, cached: str, grib_content: bytes):
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    name = '201810/20181015.cdas1.t06z.pgrbh00.grib2'
    calls = []

    def respond(request: Request) -> Response:
        calls.append(request.url.path)
        return Response(200, content=grib_content)

    with (
        download_cache(remote, work_dir=tmp_path / 'work') as cache,
        Client(transport=MockTransport(respond)) as session,
    ):
        # Keep a complete first message and the final terminator: checking only
        # the beginning/end of the file would incorrectly accept this truncation.
        broken = grib_content[:-5] + grib_content[-4:]
        if cached == 'local':
            cache.file(name).write_bytes(broken)
        with remote.write_atomic(name) as stream:
            stream.write(broken)
        downloader = download._Cfs2ReanalysisDownloader(session, cache)
        paths = downloader(date(2018, 10, 15))
        assert len(paths) == 4
        assert all(path.read_bytes() == grib_content for path in paths)
        with remote.open(name) as stream:
            assert stream.read() == grib_content
        cache.release(paths)
        assert len(downloader(date(2018, 10, 15))) == 4
        assert len(calls) == 4


@pytest.mark.parametrize('corruption', ['truncated', 'header', 'length', 'footer', 'html'])
def test_cfs2_reanalysis_rejects_invalid_response_and_recovers(
    tmp_path: Path, grib_content: bytes, corruption: str
):
    broken = {
        'truncated': grib_content[:-5] + grib_content[-4:],
        'header': grib_content + b'GRIB',
        'length': grib_content[:8] + bytes(8) + grib_content[16:],
        'footer': grib_content[:-4] + b'0000',
        'html': b'<html>upstream error</html>',
    }[corruption]
    remote = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    calls = []

    def respond(request: Request) -> Response:
        calls.append(request.url.path)
        return Response(200, content=broken if len(calls) == 1 else grib_content)

    with (
        download_cache(remote, work_dir=tmp_path / 'work') as cache,
        Client(transport=MockTransport(respond)) as session,
    ):
        downloader = download._Cfs2ReanalysisDownloader(session, cache)
        with pytest.raises(RuntimeError, match='Incomplete or invalid CFS2 GRIB download: https:'):
            downloader(date(2018, 10, 15))
        assert remote.glob('**/*.grib2') == []
        assert list(cache.root.rglob('*.grib2')) == []
        paths = downloader(date(2018, 10, 15))
        assert len(paths) == 4
        assert all(path.read_bytes() == grib_content for path in paths)
        assert len(remote.glob('**/*.grib2')) == 4
        assert len(calls) == 5


def test_cfs2_downloaders_reuse_remote_files_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grib_content: bytes
):
    fs = TrackingFileSystem(skip_instance_cache=True)
    remote = FileStore(fs, f'{tmp_path.name}/raw')
    calls = []

    def get(request: Request) -> Response:
        calls.append(str(request.url))
        return Response(200, content=grib_content)

    monkeypatch.setattr(download._Cfs2ForecastDownloader, '_min_interval', 0)
    with Client(transport=MockTransport(get)) as session:
        for _attempt in range(2):
            with download_cache(remote, 'reanalysis') as cache:
                paths = download._Cfs2ReanalysisDownloader(session, cache)(date(2000, 1, 1))
                assert paths is not None
                assert len(paths) == 4
                assert all(path.read_bytes() == grib_content for path in paths)
                assert len(remote.child('reanalysis').glob('**/*.grib2')) == 4
            with download_cache(remote, 'forecast') as cache:
                downloader = download._Cfs2ForecastDownloader(
                    session, cache, date(2000, 1, 1), date(2000, 1, 2)
                )
                downloader('flx', 'file{}{}', {})
                paths = list(cache.root.glob('*.grb2'))
                assert len(paths) == 4
                assert all(path.read_bytes() == grib_content for path in paths)
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
        lambda _proxy: nullcontext(
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


def test_cleanup_preserves_old_needed_file(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.cleanup(lambda name: name == 'needed.grib')
    assert paths['needed'].read_bytes() == b'data'


def test_cleanup_removes_old_unneeded_file(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.cleanup(lambda name: name == 'needed.grib')
    assert not paths['obsolete'].exists()


def test_cleanup_preserves_recent_file(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.cleanup(lambda _name: False)
    assert paths['recent'].read_bytes() == b'data'


def test_cleanup_preserves_cache_metadata(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.cleanup(lambda _name: False)
    assert paths['metadata'].is_file()


def test_cleanup_removes_partial_file(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.cleanup(lambda name: name == 'needed.grib')
    assert not paths['partial'].exists()


def test_release_removes_working_file(working_cache_files: tuple):
    cache, paths = working_cache_files
    cache.release([paths['needed']])
    assert not paths['needed'].exists()


def test_cleanup_and_release_preserve_local_cache(tmp_path: Path):
    with download_cache(tmp_path / 'local', 'era5', work_dir=tmp_path / 'unused') as cache:
        path = cache.file('obsolete.grib')
        path.write_bytes(b'original')
        old = time.time() - 30 * 86400
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

    @contextmanager
    def fail_upload(_self, _name):  # noqa: ANN001
        msg = 'upload interrupted'
        raise OSError(msg)
        yield  # Make failure occur when entering the upload context.

    with monkeypatch.context() as patch:
        patch.setattr(FileStore, 'write_atomic', fail_upload)
        with download_cache(remote, 'cmip6', work_dir=tmp_path / 'work') as cache:
            path = cache.file('tas_2000.nc')
            cache.write(path, b'complete provider response')
            with pytest.raises(OSError, match='upload interrupted'):
                cache.publish([path])
        assert path.read_bytes() == b'complete provider response'
        assert not remote.child('cmip6').exists('tas_2000.nc')
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
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'write_bytes', interrupt)
            with pytest.raises(OSError, match='disk write interrupted'):
                cache.write(path, b'complete response')
        assert not cache.exists('tas.nc')
        assert not list(cache.root.glob('*.part'))
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
        remote, 'cfs2', 'reanalysis', work_dir=tmp_path / 'work', target=str(store)
    ) as cache:
        done = cache.file('200001/20000101.cdas1.t00z.pgrbh00.grib2')
        needed = cache.file('200001/20000102.cdas1.t00z.pgrbh00.grib2')
    with download_cache(
        remote, 'cfs2', 'forecast', work_dir=tmp_path / 'work', target=str(store)
    ) as cache:
        forecast = cache.file('2000-01-10/flx2000010700.grb2')
        obsolete = cache.file('2000-01-09/flx2000010700.grb2')
    for path in (done, needed, forecast, obsolete):
        path.write_bytes(b'GRIB')
        os.utime(path, (old, old))

    def stop_before_network(_proxy: str | None):
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


@pytest.mark.parametrize('leading_missing', [1, 128])
@pytest.mark.parametrize('missing_status', [404, 200], ids=['not-found', 'empty-body'])
def test_cfs2_reanalysis_keeps_calendar_and_appends_missing_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leading_missing: int,
    missing_status: int,
    grib_content: bytes,
):
    begin = date(2026, 1, 1)
    monkeypatch.setattr(const, 'CFS2_REANALYSIS_FIRST_DATE', begin)
    monkeypatch.setattr(const, 'CFS2_FLX_BANDS', {'TMP': const.CFS2_FLX_BANDS['TMP']})
    store = get_store(tmp_path / 'prepared')
    requests = []
    decoded = set()

    def respond(request: Request):
        parts = request.url.path.split('/')
        day = date.fromisoformat(parts[-2])
        hour = int(parts[-1].split('.t')[1][:2])
        offset = (day - begin).days
        requests.append((offset, hour))
        if (
            offset < leading_missing
            or offset > leading_missing + 1
            or (offset == leading_missing and hour == 6)
        ):
            return Response(missing_status)
        return Response(200, content=grib_content)

    def read(path: Path, *, kind: str, product: str = 'reanalysis'):
        assert product == 'reanalysis'
        day, cycle = path.name.split('.cdas1.t')
        when = np.datetime64(f'{day[:4]}-{day[4:6]}-{day[6:]}T{cycle[:2]}:00:00')
        offset = int((when.astype('datetime64[D]') - np.datetime64(begin)).astype(int))
        hour = int(str(when)[11:13])
        decoded.add((kind, offset, hour))
        name, units = ('TMP', 'K') if kind == 'flx' else ('RH', '%')
        return xr.Dataset(
            {
                name: (
                    ('time', 'latitude', 'longitude'),
                    np.full((1, 2, 2), offset * 10 + hour, dtype=np.float32),
                    {'units': units},
                )
            },
            coords={'time': [when], 'latitude': [0, 1], 'longitude': [0, 1]},
        )

    monkeypatch.setattr(download, 'read_cfs2', read)
    end = begin + (leading_missing + 2) * const.ONE_DAY
    with (
        Client(transport=MockTransport(respond)) as session,
        download_cache(tmp_path / 'raw', 'cfs2', 'reanalysis', work_dir=tmp_path / 'work') as cache,
    ):
        download._download_cfs2_reanalysis(store, end, session, cache, work_dir=tmp_path / 'work')
        # A subsequent scheduled run has no source files at all for the new day.
        download._download_cfs2_reanalysis(
            store, end + const.ONE_DAY, session, cache, work_dir=tmp_path / 'work'
        )
    assert requests == [
        (offset, hour) for offset in range(leading_missing + 3) for hour in (0, 6, 12, 18)
    ]
    assert decoded == {
        (kind, offset, hour)
        for kind in ('flx', 'pgb')
        for offset, hours in (
            (leading_missing, (0, 12, 18)),
            (leading_missing + 1, (0, 6, 12, 18)),
        )
        for hour in hours
    }
    expected_time = np.datetime64(begin) + np.arange(leading_missing + 3).astype('timedelta64[D]')
    expected_values = np.full(leading_missing + 3, np.nan)
    expected_values[leading_missing + 1] = (leading_missing + 1) * 10 + 9
    for kind, name in (('flx', 'TMP'), ('pgb', 'RH')):
        with open_dataset(store, f'reanalysis/{kind}') as saved:
            np.testing.assert_array_equal(saved.time, expected_time)
            np.testing.assert_allclose(saved[name][:, 0, 0], expected_values, rtol=0, atol=0)
            assert saved.attrs['committed_length'] == leading_missing + 3


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
