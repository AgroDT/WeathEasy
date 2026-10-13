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
        ['correction', 'prepare-dem', '--help'],
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
                   'weatheasy.era5', 'weatheasy.query')
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


@pytest.mark.parametrize('source', ['era5', 'cfs2'])
def test_prepare_dem_writes_selected_source_grid(tmp_path: Path, source: str):
    import numpy as np
    import xarray as xr

    from weatheasy._cli import main
    from weatheasy.storage import get_store, prepare_dataset, write_batch

    weather = tmp_path / 'weather.zarr'
    grids = {
        'daily': ([55.0, 55.5, 56.0], [37.0, 37.5, 38.0]),
        'reanalysis/flx': ([40.0, 42.0, 44.0, 84.0], [18.0, 20.0, 22.0, 182.0]),
    }
    for group, (lats, lons) in grids.items():
        ds = xr.Dataset(
            {
                'TMP': (
                    ('time', 'latitude', 'longitude'),
                    np.ones((1, len(lats), len(lons)), np.float32),
                    {'units': 'degC'},
                )
            },
            coords={'time': [np.datetime64('2020-01-01')], 'latitude': lats, 'longitude': lons},
        )
        write_batch(get_store(weather), group, prepare_dataset(ds, crs='EPSG:4326'))
    native_lats = np.arange(39.75, 57.0, 0.25)
    native_lons = np.arange(17.75, 39.0, 0.25)
    heights = np.broadcast_to(
        100 + 8 * (native_lons - 19), (len(native_lats), len(native_lons))
    ).astype(np.float32)
    native = xr.Dataset(
        {'height': (('north', 'east'), heights)},
        coords={'north': native_lats, 'east': native_lons},
    )
    native = native.rio.set_spatial_dims(x_dim='east', y_dim='north').rio.write_crs('EPSG:4326')
    input_path = tmp_path / 'input.zarr'
    output_path = tmp_path / 'output.zarr'
    native.to_zarr(input_path, zarr_format=3, consolidated=True)
    arguments = [
        'correction',
        'prepare-dem',
        source,
        '--store',
        str(weather),
        '--dem-source',
        str(input_path),
        '--dem-elevation',
        'height',
        '--dem-x',
        'east',
        '--dem-y',
        'north',
        '--output',
        str(output_path),
    ]
    main(arguments, configure_logging=False)
    if source == 'era5':
        expected_lats, expected_lons = [55.0, 55.5, 56.0], [37.0, 37.5, 38.0]
    else:
        # CFS2 crops outside all four boundaries of its correction region.
        expected_lats, expected_lons = [42.0, 44.0], [20.0, 22.0]
    expected_height = np.broadcast_to(
        100 + 8 * (np.asarray(expected_lons) - 19), (len(expected_lats), len(expected_lons))
    )
    with xr.open_zarr(output_path, consolidated=True, chunks=None) as result:
        np.testing.assert_array_equal(result.latitude, expected_lats)
        np.testing.assert_array_equal(result.longitude, expected_lons)
        np.testing.assert_allclose(result.elevation, expected_height)

    native['height'] = native.height + 20
    native.to_zarr(input_path, mode='w', zarr_format=3, consolidated=True)
    with pytest.raises(FileExistsError, match='overwrite'):
        main(arguments, configure_logging=False)
    with xr.open_zarr(output_path, consolidated=True, chunks=None) as result:
        np.testing.assert_allclose(result.elevation, expected_height)
    main([*arguments, '--overwrite'], configure_logging=False)
    with xr.open_zarr(output_path, consolidated=True, chunks=None) as result:
        np.testing.assert_allclose(result.elevation, expected_height + 20)


def test_prepare_stations_parquet_and_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pandas as pd

    from weatheasy._cli import main
    from weatheasy._cli.commands import correction_prepare_stations as command

    stations = pd.DataFrame(
        {'station_id': ['a'], 'lat': [55.0], 'lon': [37.0], 'temperature': [12.0]}
    )
    source = tmp_path / 'stations.parquet'
    output = tmp_path / 'enriched.parquet'
    stations.to_parquet(source, index=False)
    calls = []

    def prepare(observations, dem, **kwargs):  # noqa: ANN001
        pd.testing.assert_frame_equal(observations, stations)
        calls.append((dem.root, kwargs))
        return observations.assign(elevation=150.0, slope_we=0.1, slope_sn=-0.2)

    monkeypatch.setattr(command, 'prepare_station_terrain', prepare)
    args = [
        'correction',
        'prepare-stations',
        '--stations',
        str(source),
        '--output',
        str(output),
        '--dem-source',
        str(tmp_path / 'dem'),
        '--dem-elevation',
        'height',
        '--dem-x',
        'longitude',
        '--dem-y',
        'latitude',
    ]
    main(args, configure_logging=False)
    expected = stations.assign(elevation=150.0, slope_we=0.1, slope_sn=-0.2)
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)
    assert calls == [(tmp_path / 'dem', {'elevation': 'height', 'x': 'longitude', 'y': 'latitude'})]
    with pytest.raises(FileExistsError, match='--overwrite'):
        main(args, configure_logging=False)
    assert len(calls) == 1
    main([*args, '--overwrite'], configure_logging=False)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)


@pytest.mark.parametrize('action', ['train', 'apply'])
@pytest.mark.parametrize('remote_weather', [True, False])
def test_correction_cli_preserves_s3_uri_and_reuses_filesystem(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, action: str, *, remote_weather: bool
):
    from io import BytesIO
    from types import SimpleNamespace

    import pandas as pd
    from zarr.storage import FsspecStore

    from weatheasy import _cli as cli
    from weatheasy._cli import _storage
    from weatheasy.correction import apply, training
    from weatheasy.file_storage import FileStore, get_file_store

    s3fs = pytest.importorskip('s3fs')
    filesystem = s3fs.S3FileSystem()
    weather = SimpleNamespace()
    received = []
    weather_uri = 's3://weather/prepared' if remote_weather else './weather'

    def artifacts_for_uri(uri, *, filesystem=None):  # noqa: ANN001
        return get_file_store(
            uri, filesystem=(filesystem or shared_filesystem) if uri.startswith('s3://') else None
        )

    shared_filesystem = filesystem
    monkeypatch.setattr(_storage, 'get_file_store', artifacts_for_uri)

    def open_weather(uri, *, filesystem):  # noqa: ANN001
        assert uri == weather_uri
        if remote_weather:
            assert filesystem.sync_fs is shared_filesystem
        else:
            assert filesystem is None
        return weather

    monkeypatch.setattr(_storage, 'get_storage', open_weather)

    def open_station_file(self: FileStore, name: str = '', mode: str = 'rb') -> BytesIO:
        assert self.fs is shared_filesystem
        assert self.root == 'observations/stations.parquet'
        assert name == ''
        assert mode == 'rb'
        stream = BytesIO()
        pd.DataFrame(
            [
                {
                    'station_id': 1,
                    'lat': 55,
                    'lon': 37,
                    'date': '2026-01-01',
                    'temperature': 8,
                    'precipitation': 3,
                    'elevation': 123.0,
                    'slope_we': 0.1,
                    'slope_sn': 0.2,
                }
            ]
        ).to_parquet(stream, index=False)
        stream.seek(0)
        return stream

    monkeypatch.setattr(FileStore, 'open', open_station_file)

    def build_artifacts(source, root, stations, artifacts, *, only):  # noqa: ANN001, ARG001
        assert stations.to_dict('records') == [
            {
                'station_id': 1,
                'lat': 55,
                'lon': 37,
                'date': '2026-01-01',
                'temperature': 8,
                'precipitation': 3,
                'elevation': 123.0,
                'slope_we': 0.1,
                'slope_sn': 0.2,
            }
        ]
        received.append((root, None, artifacts))

    def correct_store(source, root, artifacts, terrain, *, begin, end):  # noqa: ANN001, ARG001
        received.append((root, terrain, artifacts))
        return 0

    monkeypatch.setattr(training, 'build_artifacts', build_artifacts)
    monkeypatch.setattr(apply, 'correct_store', correct_store)
    arguments = [
        'correction',
        action,
        'era5',
        '--store',
        weather_uri,
        '--artifacts',
        's3://models/correction',
    ]
    if action == 'apply':
        arguments.extend(['--dem', 's3://terrain/copdem.zarr'])
    if action == 'train':
        arguments.extend(['--stations', 's3://observations/stations.parquet'])
    cli.main(arguments)
    assert len(received) == 1
    root, terrain, artifacts = received[0]
    assert root is weather
    if action == 'apply':
        assert isinstance(terrain, FsspecStore)
        assert terrain.path == 'terrain/copdem.zarr'
        assert terrain.fs.sync_fs is shared_filesystem
    else:
        assert terrain is None
    assert artifacts.fs is shared_filesystem
    assert artifacts.root == 'models/correction'
