"""Terrain preparation protects spatial aggregation and reusable feature contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

from weatheasy.correction import terrain
from weatheasy.correction.sources import CFS2, ERA5, Source
from weatheasy.storage import get_store, prepare_dataset, write_batch


def native_dem(
    *,
    reverse: bool = False,
    names: tuple[str, str, str] = ('height', 'east', 'north'),
    nodata: bool = False,
    north_gradient: float = 0,
):
    elevation, x, y = names
    lats = np.arange(53.75, 57, 0.25, dtype=np.float64)
    lons = np.arange(35.75, 39, 0.25, dtype=np.float64)
    heights = np.broadcast_to(100 + 8 * (lons[None, :] - 37), (len(lats), len(lons))).copy()
    heights += north_gradient * (lats[:, None] - 55)
    if nodata:
        heights[:, 4] = np.nan
    ds = xr.Dataset({elevation: ((y, x), heights.astype(np.float32))}, coords={y: lats, x: lons})
    ds = ds.rio.set_spatial_dims(x_dim=x, y_dim=y).rio.write_crs('EPSG:4326')
    if reverse:
        ds = ds.isel({y: slice(None, None, -1), x: slice(None, None, -1)})
    store = MemoryStore()
    ds.to_zarr(
        store,
        zarr_format=3,
        consolidated=False,
        encoding={elevation: {'chunks': (5, 5), 'grid_mapping': 'spatial_ref'}},
    )
    return store


def weather(tmp_path: Path, source: Source):
    store = get_store(tmp_path / 'weather.zarr')
    ds = xr.Dataset(
        {
            'TMP': (
                ('time', 'latitude', 'longitude'),
                np.ones((1, 3, 3), np.float32),
                {'units': 'degC'},
            )
        },
        coords={
            'time': [np.datetime64('2020-01-01')],
            'latitude': [55.0, 55.5, 56.0],
            'longitude': [37.0, 37.5, 38.0],
        },
    )
    group = 'daily' if source is ERA5 else 'reanalysis/flx'
    write_batch(store, group, prepare_dataset(ds, crs='EPSG:4326'))
    return zarr.open_group(store, mode='r', use_consolidated=False)


@pytest.mark.parametrize('source', [ERA5, CFS2], ids=['era5', 'cfs2'])
@pytest.mark.parametrize('reverse', [False, True])
def test_preparation_preserves_target_grid_and_terrain_values(
    tmp_path: Path, source: Source, *, reverse: bool
):
    native = native_dem(reverse=reverse)
    root = weather(tmp_path, source)
    output = MemoryStore()
    terrain.prepare_dem(source, root, native, output, elevation='height', x='east', y='north')
    with xr.open_zarr(output, consolidated=True, chunks=None, decode_coords='all') as ds:
        lats = np.array([55.0, 55.5, 56.0])
        lons = np.array([37.0, 37.5, 38.0])
        np.testing.assert_array_equal(ds.latitude, lats)
        np.testing.assert_array_equal(ds.longitude, lons)
        expected = np.broadcast_to(100 + 8 * (lons[None, :] - 37), (len(lats), len(lons)))
        np.testing.assert_allclose(ds.elevation, expected)
        np.testing.assert_allclose(ds.slope_sn, 0, atol=1e-10)
        expected_we = np.broadcast_to(
            8 / (6371000 * np.pi / 180 * np.cos(np.deg2rad(lats[:, None]))), expected.shape
        )
        np.testing.assert_allclose(ds.slope_we, expected_we, rtol=1e-5)
        assert ds.rio.crs == 'EPSG:4326'
        assert ds.attrs['build_status'] == 'complete'
        assert ds.elevation.dtype == np.float32
        assert ds.elevation.attrs['units'] == 'm'
        assert ds.slope_we.attrs['units'] == '1'
    group = zarr.open_group(output, mode='r', use_consolidated=False)
    assert group.get_array('elevation').chunks == (len(lats), len(lons))


def test_failed_terrain_write_remains_unpublished_and_explicit_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    native = native_dem()
    root = weather(tmp_path, ERA5)
    output = MemoryStore()
    original = MemoryStore.set
    original_sync = MemoryStore.set_sync
    interrupted = []

    def interrupt(key: str):
        if key.startswith('slope_sn/c/'):
            interrupted.append(key)
            msg = 'Interrupted terrain write'
            raise OSError(msg)

    async def write(self: MemoryStore, key: str, *args, **kwargs):
        interrupt(key)
        return await original(self, key, *args, **kwargs)

    def write_sync(self: MemoryStore, key: str, *args, **kwargs):
        interrupt(key)
        return original_sync(self, key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, 'set', write)
        patch.setattr(MemoryStore, 'set_sync', write_sync)
        with pytest.raises(OSError, match='Interrupted terrain write'):
            terrain.prepare_dem(ERA5, root, native, output, elevation='height', x='east', y='north')
        assert interrupted
        group = zarr.open_group(output, mode='r', use_consolidated=False)
        assert group.attrs['build_status'] == 'writing'
        assert group.metadata.consolidated_metadata is None

    terrain.prepare_dem(
        ERA5, root, native, output, elevation='height', x='east', y='north', overwrite=True
    )
    with xr.open_zarr(output, consolidated=True, chunks=None) as result:
        assert result.attrs['build_status'] == 'complete'
        np.testing.assert_allclose(result.elevation, [[100, 104, 108]] * 3, rtol=1e-6)
        np.testing.assert_allclose(result.slope_sn, 0, atol=1e-10)
        distances = 6_371_000 * np.pi / 180 * np.cos(np.deg2rad([55, 55.5, 56]))
        np.testing.assert_allclose(
            result.slope_we, np.repeat((8 / distances)[:, None], 3, axis=1), rtol=1e-5
        )


@pytest.mark.parametrize('reverse', [False, True])
def test_station_features_preserve_observations_and_native_slopes(*, reverse: bool):
    observations = pd.DataFrame(
        {
            'station_id': [2, 1, 2],
            'lat': [55.0, 55.25, 55.0],
            'lon': [37.0, 37.5, 37.0],
            'temperature': [3.0, 5.0, 7.0],
            'elevation': [-1.0, -1.0, -1.0],
        },
        index=[9, 3, 5],
    )
    original = observations.copy()
    result = terrain.prepare_station_terrain(
        observations,
        native_dem(reverse=reverse, north_gradient=-12),
        elevation='height',
        x='east',
        y='north',
    )
    pd.testing.assert_frame_equal(observations, original)
    pd.testing.assert_frame_equal(
        result.drop(columns=['elevation', 'slope_we', 'slope_sn']),
        original.drop(columns='elevation'),
    )
    np.testing.assert_allclose(result.elevation, [100, 101, 100])
    # The plane loses 3m over 0.25 degrees north: a negative, nonzero slope.
    north_distance = 2 * np.pi * 6_371_000 * (0.25 / 360)
    np.testing.assert_allclose(result.slope_sn, -3 / north_distance, rtol=1e-5)
    expected = 8 / (6371000 * np.pi / 180 * np.cos(np.deg2rad(result.lat)))
    np.testing.assert_allclose(result.slope_we, expected, rtol=1e-5)


def test_missing_native_cells_are_zero_in_full_cell_average(tmp_path: Path):
    root = weather(tmp_path, ERA5)
    native = native_dem(nodata=True)
    output = MemoryStore()
    terrain.prepare_dem(ERA5, root, native, output, elevation='height', x='east', y='north')
    ds = xr.open_zarr(output, chunks=None, consolidated=True)
    # West cell spans 36.75..37.25: west native pixel contributes half its area,
    # centre pixel contributes its full area; the other half is the east pixel.
    # Missing west native pixel (98m) contributes zero, not a reduced denominator.
    np.testing.assert_allclose(ds.elevation[:, 0], 75.5)
    ds.close()


def test_area_weighting_uses_partial_native_footprints():
    ds = xr.Dataset(
        {
            'elevation': (
                ('y', 'x'),
                np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]], np.float32),
            )
        },
        coords={'y': [45.0, 55.0, 65.0], 'x': [0.0, 10.0]},
    )
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:4326')
    native = MemoryStore()
    ds.to_zarr(
        native,
        consolidated=False,
        zarr_format=3,
        encoding={'elevation': {'chunks': (2, 1), 'grid_mapping': 'spatial_ref'}},
    )
    with terrain._native(native, 'elevation', 'x', 'y') as field:
        actual = terrain._aggregate(field, np.array([50.0, 60.0]), np.array([2.0, 8.0]))
    # Target latitude footprints45..55 and55..65 split native40..50,50..60,60..70.
    sin = lambda lat: np.sin(np.deg2rad(lat))  # noqa: E731
    first = 100 * (sin(55) - sin(50)) / (sin(55) - sin(45))
    second = (100 * (sin(60) - sin(55)) + 200 * (sin(65) - sin(60))) / (sin(65) - sin(55))
    np.testing.assert_allclose(actual, [[first, first], [second, second]], rtol=1e-6)
    assert not np.allclose(actual[:, 0], [50, 150])


def test_preparation_rejects_source_coverage_gap():
    with (
        pytest.raises(ValueError, match='does not cover'),
        terrain._native(native_dem(), 'height', 'east', 'north') as field,
    ):
        terrain._aggregate(field, np.array([52.0, 55.0]), np.array([37.0, 38.0]))


def test_global_dem_means_wrap_cell_at_180():
    lons = np.arange(-179.0, 180.0, 2.0)
    heights = np.ones((2, len(lons)), np.float32) * 30
    heights[:, 0] = 100
    heights[:, -1] = 200
    ds = xr.Dataset({'elevation': (('y', 'x'), heights)}, coords={'y': [55.0, 57.0], 'x': lons})
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs('EPSG:4326')
    native = MemoryStore()
    ds.to_zarr(
        native,
        consolidated=False,
        zarr_format=3,
        encoding={'elevation': {'chunks': (2, 10), 'grid_mapping': 'spatial_ref'}},
    )
    with terrain._native(native, 'elevation', 'x', 'y') as field:
        actual = terrain._aggregate(field, np.array([55.0, 57.0]), np.array([178.0, 180.0]))
    # 177..179 covers half of177-centre30m and179-centre200m;
    # 179..181 straddles the seam, half200m and half100m.
    np.testing.assert_allclose(actual, [[115.0, 150.0], [115.0, 150.0]])


@pytest.mark.parametrize('target', ['native', 'weather', 'weather_child', 'weather_parent'])
def test_prepare_dem_preserves_inputs_on_overwrite_collision(tmp_path: Path, target: str):
    native = native_dem()
    root = weather(tmp_path, ERA5)
    with xr.open_zarr(native, chunks=None, consolidated=False, decode_coords='all') as ds:
        source_before = ds.load().copy(deep=True)
    weather_before = np.asarray(
        zarr.open_group(root.store, mode='r', use_consolidated=False).get_array('daily/TMP')[:]
    ).copy()
    outputs = {
        'native': native.with_read_only(read_only=False),
        'weather': get_store(tmp_path / 'weather.zarr' / '..' / 'weather.zarr'),
        'weather_child': get_store(tmp_path / 'weather.zarr' / 'daily'),
        'weather_parent': get_store(tmp_path),
    }
    output = outputs[target]
    with pytest.raises(ValueError, match='must be separate'):
        terrain.prepare_dem(
            ERA5, root, native, output, elevation='height', x='east', y='north', overwrite=True
        )
    with xr.open_zarr(native, chunks=None, consolidated=False, decode_coords='all') as ds:
        xr.testing.assert_identical(ds.load(), source_before)
    np.testing.assert_array_equal(
        zarr.open_group(root.store, mode='r', use_consolidated=False).get_array('daily/TMP')[:],
        weather_before,
    )


def test_prepare_dem_cli_preserves_weather_on_output_collision(tmp_path: Path):
    from weatheasy._cli import main

    root = weather(tmp_path, ERA5)
    before = np.asarray(root.get_array('daily/TMP')[:]).copy()
    with pytest.raises(ValueError, match='must be separate'):
        main(
            [
                'correction',
                'prepare-dem',
                'era5',
                '--store',
                str(tmp_path / 'weather.zarr'),
                '--dem-source',
                str(tmp_path / 'unread-native.zarr'),
                '--output',
                str(tmp_path / 'weather.zarr'),
                '--overwrite',
            ]
        )
    np.testing.assert_array_equal(root.get_array('daily/TMP')[:], before)


def test_prepare_dem_rejects_same_remote_target_before_reading(tmp_path: Path):
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from fsspec.implementations.memory import MemoryFileSystem
    from zarr.storage import FsspecStore

    filesystem = MemoryFileSystem()
    left = FsspecStore(AsyncFileSystemWrapper(fs=filesystem, asynchronous=True), path='bucket/dem')
    right = FsspecStore(
        AsyncFileSystemWrapper(fs=filesystem, asynchronous=True), path='bucket/dem/'
    )
    with pytest.raises(ValueError, match='must be separate'):
        terrain.prepare_dem(ERA5, weather(tmp_path, ERA5), left, right, overwrite=True)
