from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path


# Help must be available even when all command dependencies are unavailable.
@pytest.mark.parametrize(
    'arguments',
    [
        ['--help'],
        ['--version'],
        ['query', 'cmip6', '--help'],
        ['download', 'era5', '--help'],
        ['dem', '--help'],
        ['correction', 'train', '--help'],
        ['emdna', 'temperature', '--help'],
        ['list-vars'],
    ],
)
def test_cli_metadata_does_not_import_command_dependencies(arguments: list[str]):
    script = """
import importlib.abc
import runpy
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        packages = {'pandas', 'xarray', 'rioxarray', 'rasterio', 'zarr',
                    'scipy', 'fsspec', 's3fs', 'cdsapi', 'cfgrib', 'eccodes',
                    'xgboost', 'sklearn', 'joblib', 'fastapi', 'uvicorn'}
        modules = ('weatheasy.correction', 'weatheasy.emdna', 'weatheasy.download',
                   'weatheasy.era5', 'weatheasy.build_copdem', 'weatheasy.query')
        if fullname.split('.')[0] in packages or fullname.startswith(modules):
            raise AssertionError('Unexpected command dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['weatheasy', *sys.argv[1:]]
runpy.run_module('weatheasy', run_name='__main__')
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, '-c', script, *arguments], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ('source', 'group', 'variable', 'units'),
    [
        ('cmip6', 'tas', 'tas', 'K'),
        ('cfs2', 'reanalysis/flx', 'TMP', 'degC'),
        ('era5', 'daily', 'TMP', 'degC'),
    ],
)
def test_query_local_store_without_optional_imports(
    tmp_path: Path, source: str, group: str, variable: str, units: str
):
    import numpy as np
    import xarray as xr

    from weatheasy.storage import get_store, prepare_dataset, set_group_attrs, write_batch

    store = tmp_path / 'weather.zarr'
    if source == 'cfs2':
        set_group_attrs(get_store(store), '', updated='2000-01-05')
    write_batch(
        get_store(store),
        group,
        prepare_dataset(
            xr.Dataset(
                {
                    variable: (
                        ('time', 'latitude', 'longitude'),
                        np.array([[[280.25, 290.0], [310.0, 320.0]]]),
                        {'units': units},
                    )
                },
                coords={
                    'time': [np.datetime64('2000-01-01')],
                    'latitude': [55.0, 56.0],
                    'longitude': [37.0, 38.0],
                },
            ),
            crs='EPSG:4326',
        ),
    )
    script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        packages = {'s3fs', 'cdsapi', 'cfgrib', 'eccodes', 'xgboost',
                    'sklearn', 'joblib', 'fastapi', 'uvicorn'}
        modules = ('weatheasy.correction', 'weatheasy.emdna', 'weatheasy.download')
        if fullname.split('.')[0] in packages or fullname.startswith(modules):
            raise AssertionError('Unexpected command dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
from weatheasy._cli import main
main(sys.argv[1:])
print('stdout-still-open')
"""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            '-c',
            script,
            'query',
            source,
            '--store',
            str(store),
            '2000-01-01',
            '2000-01-01',
            '55',
            '37',
            variable,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f'DATE,{variable}\n2000-01-01,280.25\nstdout-still-open\n'


def test_query_s3_result_reuses_store_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import numpy as np
    import xarray as xr
    from fsspec.implementations.memory import MemoryFileSystem

    from weatheasy._cli import _storage, main
    from weatheasy.file_storage import FileStore, get_file_store
    from weatheasy.storage import get_store, prepare_dataset, write_batch

    weather = get_store(tmp_path / 'weather.zarr')
    write_batch(
        weather,
        'tas',
        prepare_dataset(
            xr.Dataset(
                {
                    'tas': (
                        ('time', 'latitude', 'longitude'),
                        np.array([[[280.25, 290], [310, 320]]]),
                        {'units': 'K'},
                    )
                },
                coords={
                    'time': [np.datetime64('2000-01-01')],
                    'latitude': [55, 56],
                    'longitude': [37, 38],
                },
            ),
            crs='EPSG:4326',
        ),
    )
    fs = MemoryFileSystem()
    locations = []

    def files(uri, *, filesystem=None):  # noqa: ANN001
        locations.append((uri, filesystem))
        return FileStore(fs, uri[5:]) if uri.startswith('s3://') else get_file_store(uri)

    opened = []

    def store(uri, *, filesystem=None):  # noqa: ANN001
        assert uri == 's3://weather/prepared'
        assert filesystem is not None
        assert filesystem.sync_fs is fs
        opened.append(uri)
        return weather

    monkeypatch.setattr(_storage, 'get_file_store', files)
    monkeypatch.setattr(_storage, 'get_store', store)
    main(
        [
            'query',
            'cmip6',
            '--store',
            's3://weather/prepared',
            '-o',
            's3://results/query.csv',
            '2000-01-01',
            '2000-01-01',
            '55',
            '37',
            'tas',
        ],
        configure_logging=False,
    )
    assert opened == ['s3://weather/prepared']
    assert locations == [('s3://weather/prepared', None), ('s3://results/query.csv', fs)]
    assert fs.cat('results/query.csv') == b'DATE,tas\n2000-01-01,280.25\n'


@pytest.mark.parametrize('source', ['cfs2', 'cmip6'])
def test_download_dispatches_prepared_store_and_local_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
):
    from zarr.storage import LocalStore

    from weatheasy import download
    from weatheasy._cli import main

    calls = []

    def record(store, download_dir, **kwargs):  # noqa: ANN001
        assert isinstance(store, LocalStore)
        assert kwargs == {'work_dir': tmp_path / 'work', 'work_max_age_days': 3}
        calls.append((store.root, download_dir.root))

    monkeypatch.setattr(download, f'download_{source}_data', record)
    # Handler imports occur during this command, avoiding dependency imports during help.
    main(
        [
            'download',
            source,
            '--store',
            str(tmp_path / 'weather'),
            '--download-cache',
            str(tmp_path / 'downloads'),
            '--work-dir',
            str(tmp_path / 'work'),
            '--work-max-age-days',
            '3',
        ],
        configure_logging=False,
    )
    assert calls == [(tmp_path / 'weather', str(tmp_path / 'downloads'))]


def test_dem_dispatches_spatial_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from zarr.storage import LocalStore

    from weatheasy import build_copdem
    from weatheasy._cli import main

    calls = []

    def record(bbox, resolution, store):  # noqa: ANN001
        assert isinstance(store, LocalStore)
        calls.append((bbox, resolution, store.root))

    monkeypatch.setattr(build_copdem, 'build_copdem', record)
    main(
        [
            'dem',
            '--bbox',
            '56',
            '37',
            '55',
            '38',
            '--resolution',
            '0.2',
            '-o',
            str(tmp_path / 'dem.zarr'),
        ],
        configure_logging=False,
    )
    assert calls == [((56, 37, 55, 38), 0.2, tmp_path / 'dem.zarr')]


@pytest.mark.parametrize('source', ['cfs2', 'cmip6', 'era5'])
def test_download_cli_preserves_s3_cache_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
):
    from fsspec.implementations.memory import MemoryFileSystem

    from weatheasy import download, era5
    from weatheasy._cli import _storage, main
    from weatheasy.file_storage import FileStore

    fs = MemoryFileSystem()
    locations = []

    def files(uri: str, *, filesystem=None):  # noqa: ANN001
        locations.append(uri)
        assert filesystem is None or filesystem is fs
        return FileStore(fs, uri.removeprefix('s3://'))

    calls = []

    def prepare(_store, cache, **_kwargs):  # noqa: ANN001
        calls.append(cache)

    monkeypatch.setattr(_storage, 'get_file_store', files)
    monkeypatch.setattr(era5 if source == 'era5' else download, f'download_{source}_data', prepare)
    main(
        [
            'download',
            source,
            '--store',
            str(tmp_path / 'prepared'),
            '--download-cache',
            's3://weather/raw',
        ],
        configure_logging=False,
    )
    assert locations == ['s3://weather/raw', 's3://weather/raw']
    assert len(calls) == 1
    assert calls[0].root == 'weather/raw'
    assert calls[0].fs is fs
