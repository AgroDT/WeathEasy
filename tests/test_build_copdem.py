from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from zarr.storage import FsspecStore, MemoryStore

from weatheasy import build_copdem
from weatheasy.correction.features import load_dem
from weatheasy.storage import get_store


@pytest.mark.parametrize('in_memory', [False, True])
def test_build_publishes_spatial_terrain_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    in_memory: bool,
):
    tile = tmp_path / 'tile.tif'
    with rasterio.open(
        tile,
        'w',
        driver='GTiff',
        width=2,
        height=2,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=from_origin(10, 61, 0.5, 0.5),
    ) as src:
        src.write(np.array([[30, 40], [10, 20]], np.float32), 1)

    original_read = build_copdem._read_tile_block

    def read_tile(url, cells):  # noqa: ANN001
        return original_read(str(tile), cells) if 'E010_' in url else None

    monkeypatch.setattr(build_copdem, '_read_tile_block', read_tile)
    store = MemoryStore() if in_memory else get_store(tmp_path / 'dem.zarr')
    build_copdem.build_copdem((61, 10, 60, 12), 0.5, store)
    elevation = np.array([[10, 20, 0, 0], [30, 40, 0, 0]], np.float32)
    with xr.open_zarr(store, consolidated=True, chunks=None, decode_coords='all') as ds:
        np.testing.assert_array_equal(ds.latitude, [60.25, 60.75])
        np.testing.assert_array_equal(ds.longitude, [10.25, 10.75, 11.25, 11.75])
        np.testing.assert_array_equal(ds.elevation, elevation)
        assert ds.elevation.dims == ('latitude', 'longitude')
        assert ds.elevation.attrs['units'] == 'm'
        assert ds.slope_we.attrs['units'] == ds.slope_sn.attrs['units'] == '1'
        assert ds.latitude.attrs['standard_name'] == 'latitude'
        assert ds.longitude.attrs['standard_name'] == 'longitude'
        assert ds.latitude.attrs['axis'] == 'Y'
        assert ds.longitude.attrs['axis'] == 'X'
        assert ds.rio.crs.to_epsg() == 4326
        assert ds.rio.bounds() == (10, 60, 12, 61)
        assert ds.elevation.encoding['grid_mapping'] == 'spatial_ref'
        np.testing.assert_allclose(ds.slope_sn, np.array([[20, 20, 0, 0]] * 2) / 55660)
        # At the western edge, 10 metres gained eastward over half a degree.
        np.testing.assert_allclose(
            ds.slope_we[:, 0],
            10 / (55660 * np.cos(np.deg2rad([60.25, 60.75]))),
            rtol=1e-6,
        )

    tile.unlink()
    terrain = load_dem(store)
    np.testing.assert_array_equal(terrain['elevation'], elevation)
    np.testing.assert_array_equal(terrain['lats'], [60.25, 60.75])
    np.testing.assert_array_equal(terrain['lons'], [10.25, 10.75, 11.25, 11.75])


@pytest.mark.parametrize('action', ['train', 'apply'])
@pytest.mark.parametrize('remote_weather', [True, False])
def test_correction_cli_preserves_s3_uri_and_reuses_filesystem(
    monkeypatch: pytest.MonkeyPatch, action: str, *, remote_weather: bool
):
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
                }
            ]
        ).to_parquet(stream, index=False)
        stream.seek(0)
        return stream

    monkeypatch.setattr(FileStore, 'open', open_station_file)

    def build_artifacts(source, root, stations, terrain, artifacts, *, only):  # noqa: ANN001, ARG001
        assert stations.to_dict('records') == [
            {
                'station_id': 1,
                'lat': 55,
                'lon': 37,
                'date': '2026-01-01',
                'temperature': 8,
                'precipitation': 3,
            }
        ]
        received.append((root, terrain, artifacts))

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
        '--dem',
        's3://terrain/copdem.zarr',
        '--artifacts',
        's3://models/correction',
    ]
    if action == 'train':
        arguments.extend(['--stations', 's3://observations/stations.parquet'])
    cli.main(arguments)
    assert len(received) == 1
    root, terrain, artifacts = received[0]
    assert root is weather
    assert isinstance(terrain, FsspecStore)
    assert terrain.path == 'terrain/copdem.zarr'
    assert terrain.fs.sync_fs is shared_filesystem
    assert artifacts.fs is shared_filesystem
    assert artifacts.root == 'models/correction'


@pytest.mark.parametrize('stage', ['tile', 'slopes'])
def test_resume_retries_unfinished_chunk_and_skips_completed_chunk(
    monkeypatch: pytest.MonkeyPatch, stage: str
):
    import zarr

    store = MemoryStore()
    reads = []
    original_slopes = build_copdem._slopes

    def read(url: str, cells: int):
        reads.append(url)
        if 'E010_' in url:
            return np.array([[10, 20], [30, 40]], np.float32)
        if 'E011_' in url:
            return None
        if stage == 'tile':
            msg = 'interrupted tile'
            raise RuntimeError(msg)
        return np.full((cells, cells), 50, np.float32)

    def interrupted_slopes(*_args):
        msg = 'interrupted slopes'
        raise RuntimeError(msg)

    monkeypatch.setattr(build_copdem, '_read_tile_block', read)
    if stage == 'slopes':
        monkeypatch.setattr(build_copdem, '_slopes', interrupted_slopes)
    with pytest.raises(RuntimeError, match='interrupted'):
        build_copdem.build_copdem((61, 10, 60, 13), 0.5, store)
    group = zarr.open_group(store)
    np.testing.assert_array_equal(
        group['tile_status'][:], [[0, 0, 0] if stage == 'tile' else [1, 2, 1]]
    )
    with pytest.raises(ValueError, match='incomplete'):
        load_dem(store)
    reads.clear()

    def resume_read(url: str, cells: int):
        reads.append(url)
        if 'E010_' in url:
            return np.array([[10, 20], [30, 40]], np.float32)
        if 'E011_' in url:
            return None
        return np.full((cells, cells), 50, np.float32)

    monkeypatch.setattr(build_copdem, '_read_tile_block', resume_read)
    monkeypatch.setattr(build_copdem, '_slopes', original_slopes)
    build_copdem.build_copdem((61, 10, 60, 13), 0.5, store)
    assert len(reads) == (3 if stage == 'tile' else 0)
    assert zarr.open_group(store).attrs['build_status'] == 'complete'
    np.testing.assert_array_equal(group['tile_status'][:], [[1, 2, 1]])
    np.testing.assert_array_equal(
        load_dem(store)['elevation'], [[10, 20, 0, 0, 50, 50], [30, 40, 0, 0, 50, 50]]
    )
    reads.clear()
    build_copdem.build_copdem((61, 10, 60, 13), 0.5, store)
    assert reads == []
    with pytest.raises(ValueError, match='parameters differ'):
        build_copdem.build_copdem((61, 10, 60, 13), 0.25, store)


@pytest.mark.parametrize(
    ('error', 'missing'),
    [
        ('HTTP response code: 404', True),
        ('HTTP response code: 403', False),
        ('invalid TIFF', False),
        ('tile404.tif: invalid TIFF', False),
    ],
)
def test_remote_read_only_treats_404_as_missing(
    monkeypatch: pytest.MonkeyPatch, error: str, *, missing: bool
):
    def fail(url: str):
        assert url == '/vsicurl/https://example.com/tile.tif'
        raise rasterio.errors.RasterioIOError(error)

    monkeypatch.setattr(rasterio, 'open', fail)
    if missing:
        assert build_copdem._read_tile_block('https://example.com/tile.tif', 2) is None
    else:
        with pytest.raises(rasterio.errors.RasterioIOError, match=error):
            build_copdem._read_tile_block('https://example.com/tile.tif', 2)


@pytest.mark.parametrize('stage', ['initialization', 'elevation_write'])
def test_resume_after_interrupted_store_write(monkeypatch: pytest.MonkeyPatch, stage: str):
    import zarr

    store = MemoryStore()
    original_initialize = build_copdem._initialize_zarr
    original_setitem = zarr.Array.__setitem__

    def interrupted_initialize(*args):
        original_initialize(*args)
        msg = 'interrupted initialization'
        raise RuntimeError(msg)

    def interrupted_setitem(self, key, value):  # noqa: ANN001
        original_setitem(self, key, value)
        if self.name == '/elevation' and reads:
            msg = 'interrupted elevation write'
            raise RuntimeError(msg)

    reads = []

    def read(url: str, cells: int):
        reads.append(url)
        return np.full((cells, cells), 25, np.float32)

    monkeypatch.setattr(build_copdem, '_read_tile_block', read)
    if stage == 'initialization':
        monkeypatch.setattr(build_copdem, '_initialize_zarr', interrupted_initialize)
    else:
        monkeypatch.setattr(zarr.Array, '__setitem__', interrupted_setitem)
    with pytest.raises(RuntimeError, match='interrupted'):
        build_copdem.build_copdem((61, 10, 60, 11), 0.5, store)
    group = zarr.open_group(store)
    np.testing.assert_array_equal(group['tile_status'][:], [[0]])
    assert group.attrs['build_status'] != 'complete'
    monkeypatch.setattr(build_copdem, '_initialize_zarr', original_initialize)
    monkeypatch.setattr(zarr.Array, '__setitem__', original_setitem)
    reads.clear()
    build_copdem.build_copdem((61, 10, 60, 11), 0.5, store)
    assert len(reads) == 1
    np.testing.assert_array_equal(load_dem(store)['elevation'], [[25, 25], [25, 25]])


def test_initialization_does_not_write_field_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import zarr

    path = tmp_path / 'dem.zarr'
    store = get_store(path)

    def interrupted_read(_url: str, _cells: int):
        msg = 'stop before first tile'
        raise RuntimeError(msg)

    monkeypatch.setattr(build_copdem, '_read_tile_block', interrupted_read)
    with pytest.raises(RuntimeError, match='stop before first tile'):
        build_copdem.build_copdem((62, 10, 60, 12), 0.5, store)
    files = {str(file.relative_to(path)) for file in path.rglob('*') if file.is_file()}
    for name in ('elevation', 'slope_we', 'slope_sn', 'tile_status'):
        assert {file for file in files if file.startswith(f'{name}/')} == {f'{name}/zarr.json'}
    assert any(file.startswith('latitude/c/') for file in files)
    assert any(file.startswith('longitude/c/') for file in files)
    group = zarr.open_group(store)
    np.testing.assert_array_equal(group['tile_status'][:], np.zeros((2, 2), dtype=np.uint8))
    assert np.isnan(group['elevation'][:]).all()
    assert group.attrs['build_status'] == 'building'
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_coords='all') as ds:
        assert ds.rio.crs.to_epsg() == 4326
        assert ds.rio.bounds() == (10, 60, 12, 62)


def test_grouped_chunks_are_written_once_and_resumed_as_whole_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from io import StringIO

    import zarr

    monkeypatch.setattr(build_copdem, '_CHUNK_TILES', 2)
    path = tmp_path / 'dem.zarr'
    store = get_store(path)
    reads = []
    writes = []
    original_setitem = zarr.Array.__setitem__
    original_progress = build_copdem.tqdm
    bars = []

    def record_progress(**kwargs):
        kwargs.update(disable=False, file=StringIO())
        bar = original_progress(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(build_copdem, 'tqdm', record_progress)
    interrupted = True

    def read(url: str, cells: int):
        reads.append(url)
        if 'E012_' in url and interrupted:
            msg = 'interrupted second chunk'
            raise RuntimeError(msg)
        values = {
            'N60_00_E010': 10,
            'N60_00_E011': None,
            'N60_00_E012': 30,
            'N61_00_E010': 40,
            'N61_00_E011': 50,
            'N61_00_E012': 60,
        }
        tile = url.rsplit('/', 1)[-1].removeprefix('Copernicus_DSM_COG_30_')
        value = values[tile.removesuffix('_00_DEM.tif')]
        return None if value is None else np.full((cells, cells), value, np.float32)

    def track_write(self, key, value):  # noqa: ANN001
        if self.name in ('/elevation', '/tile_status'):
            writes.append((self.name, value.shape))
        original_setitem(self, key, value)

    monkeypatch.setattr(build_copdem, '_read_tile_block', read)
    monkeypatch.setattr(zarr.Array, '__setitem__', track_write)
    with pytest.raises(RuntimeError, match='interrupted second chunk'):
        build_copdem.build_copdem((62, 10, 60, 13), 0.5, store)
    group = zarr.open_group(store)
    np.testing.assert_array_equal(group['tile_status'][:], [[1, 2, 0], [1, 1, 0]])
    assert writes == [('/elevation', (4, 4)), ('/tile_status', (2, 2))]
    assert [(bar.desc, bar.total, bar.n) for bar in bars] == [
        ('Chunks', 2, 1),
        ('Tiles', 4, 4),
        ('Tiles', 2, 0),
    ]
    assert all(bar.disable for bar in bars)  # Context managers close bars on failure.
    bars.clear()
    interrupted = False
    reads.clear()
    writes.clear()
    build_copdem.build_copdem((62, 10, 60, 13), 0.5, store)
    assert [(bar.desc, bar.total, bar.initial, bar.n) for bar in bars] == [
        ('Chunks', 2, 1, 2),
        ('Tiles', 2, 0, 2),
    ]
    assert bars[1].postfix == 'Copernicus_DSM_COG_30_N61_00_E012_00_DEM'
    assert len(reads) == 2
    assert all('E012_' in url for url in reads)
    assert writes == [('/elevation', (4, 2)), ('/tile_status', (2, 1))]
    np.testing.assert_array_equal(
        load_dem(store)['elevation'],
        [
            [10, 10, 0, 0, 30, 30],
            [10, 10, 0, 0, 30, 30],
            [40, 40, 50, 50, 60, 60],
            [40, 40, 50, 50, 60, 60],
        ],
    )
    assert len(list((path / 'elevation' / 'c').rglob('*/*'))) == 2
    assert len(list((path / 'tile_status' / 'c').rglob('*/*'))) == 2
