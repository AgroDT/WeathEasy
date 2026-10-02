from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

    def fetch_tile(tlat, tlon, cache_dir, session):  # noqa: ANN001, ARG001
        assert cache_dir == tmp_path / 'tiles'
        return tile if tlon == 10 else None

    monkeypatch.setattr(build_copdem, '_fetch_tile', fetch_tile)
    store = MemoryStore() if in_memory else get_store(tmp_path / 'dem.zarr')
    build_copdem.build_copdem((61, 10, 60, 12), 0.5, store, tmp_path / 'tiles')
    elevation = np.array([[10, 20, 0, 0], [30, 40, 0, 0]], np.float32)
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_coords='all') as ds:
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
    (tmp_path / 'tiles').rmdir()
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
        assert self.root == 'observations/stations.csv'
        assert name == ''
        assert mode == 'rb'
        return BytesIO(b'station_id,lat,lon,date,temperature\n1,55,37,2026-01-01,8\n')

    monkeypatch.setattr(FileStore, 'open', open_station_file)

    def build_artifacts(source, root, stations, terrain, artifacts, *, only):  # noqa: ANN001, ARG001
        assert stations.to_dict('records') == [
            {'station_id': 1, 'lat': 55, 'lon': 37, 'date': '2026-01-01', 'temperature': 8}
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
        arguments.extend(['--stations', 's3://observations/stations.csv'])
    cli.main(arguments)
    assert len(received) == 1
    root, terrain, artifacts = received[0]
    assert root is weather
    assert isinstance(terrain, FsspecStore)
    assert terrain.path == 'terrain/copdem.zarr'
    assert terrain.fs.sync_fs is shared_filesystem
    assert artifacts.fs is shared_filesystem
    assert artifacts.root == 'models/correction'
