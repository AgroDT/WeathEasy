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


def test_correction_cli_preserves_s3_uri_and_reuses_filesystem(
    monkeypatch: pytest.MonkeyPatch,
):
    from weatheasy.correction import __main__ as cli, training

    s3fs = pytest.importorskip('s3fs')
    filesystem = s3fs.S3FileSystem(asynchronous=True)
    weather = SimpleNamespace(store=get_store('s3://weather/prepared', filesystem=filesystem))
    monkeypatch.setattr(cli, 'get_storage', lambda _uri: weather)
    received = []

    def build_artifacts(source, root, stations, terrain, artifacts, *, only):  # noqa: ANN001, ARG001
        received.append((root, terrain))

    monkeypatch.setattr(training, 'build_artifacts', build_artifacts)
    cli.main(
        [
            'train',
            'era5',
            '--store',
            's3://weather/prepared',
            '--dem',
            's3://terrain/copdem.zarr',
            '--stations',
            'stations.csv',
        ],
    )
    assert len(received) == 1
    root, terrain = received[0]
    assert root is weather
    assert isinstance(terrain, FsspecStore)
    assert terrain.path == 'terrain/copdem.zarr'
    assert terrain.fs is filesystem
