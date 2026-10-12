from __future__ import annotations

import builtins
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from rasterio import CRS
from zarr.storage import FsspecStore, LocalStore, MemoryStore, StoreLike

from weatheasy.storage import (
    StorageError,
    delete_group,
    get_store,
    open_dataset,
    prepare_dataset,
    remove_group_attr,
    select_point,
    set_group_attrs,
    write_batch,
    write_batches,
)
from weatheasy.util import get_storage


TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


def field(
    *,
    latitude: tuple[float, ...] = (55.125, 55.375),
    longitude: tuple[float, ...] = (0.125, 0.375),
    day: str = '2026-01-01',
):
    values = np.arange(len(latitude) * len(longitude), dtype='float32').reshape(
        1, len(latitude), len(longitude)
    )
    ds = xr.Dataset(
        {'TMP': (('time', 'latitude', 'longitude'), values, {'units': 'K'})},
        coords={
            'latitude': list(latitude),
            'longitude': list(longitude),
            'time': [np.datetime64(day)],
        },
    )
    return prepare_dataset(ds, crs='EPSG:4326')


def test_round_trip_metadata_nan_and_direct_xarray(tmp_path: Path):
    ds = field()
    ds.TMP.data[0, 0, 0] = np.nan
    path = tmp_path / 'weather.zarr'
    write_batch(get_store(path), 'ACCESS-CM2/historical/r1i1p1f1', ds, tile_size=2)
    with xr.open_zarr(
        path,
        group='ACCESS-CM2/historical/r1i1p1f1',
        consolidated=True,
        chunks=None,
        decode_coords='all',
    ) as result:
        assert result.attrs['write_state'] == 'complete'
        assert result.rio.crs == ds.rio.crs == CRS.from_epsg(4326)
        assert result.spatial_ref.attrs['grid_mapping_name'] == 'latitude_longitude'
        assert result.spatial_ref.attrs['semi_major_axis'] == 6378137
        assert result.spatial_ref.attrs['inverse_flattening'] == pytest.approx(298.257223563)
        assert result.latitude.attrs['standard_name'] == 'latitude'
        assert result.longitude.attrs['units'] == 'degrees_east'
        assert result.TMP.encoding['grid_mapping'] == 'spatial_ref'
        assert 'GeoTransform' not in result.spatial_ref.attrs
        assert result.TMP.attrs['units'] == 'K'
        np.testing.assert_array_equal(result.time, ds.time)
        np.testing.assert_array_equal(result.latitude, ds.latitude)
        np.testing.assert_array_equal(result.longitude, ds.longitude)
        np.testing.assert_array_equal(result.TMP, ds.TMP)
    root = zarr.open_group(path)
    arr = root.get_array('ACCESS-CM2/historical/r1i1p1f1/TMP')
    assert arr.metadata.zarr_format == 3
    assert arr.metadata.dimension_names == ('time', 'latitude', 'longitude')
    assert arr.chunks == (1, 2, 2)
    # A new process importing only xarray can read the artifact.
    script = """import xarray as xr, sys
x=xr.open_zarr(sys.argv[1],group='ACCESS-CM2/historical/r1i1p1f1',chunks=None,consolidated=True,decode_coords='all')
assert x.TMP.sel(latitude=55.375,longitude=0.375).item()==3
assert 'weatheasy' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script, str(path)], check=True)  # noqa: S603


def test_append_refreshes_consolidated_shape_and_commit_marker():
    store = MemoryStore()
    write_batch(store, 'forecast/flx', field())
    write_batch(store, 'forecast/flx', field(day='2026-01-02'))
    with xr.open_zarr(
        store, group='forecast/flx', consolidated=True, chunks=None, decode_coords='all'
    ) as ds:
        assert ds.sizes['time'] == ds.attrs['committed_length'] == 2
        assert ds.attrs['write_state'] == 'complete'
        np.testing.assert_array_equal(
            ds.time, np.array(['2026-01-01', '2026-01-02'], dtype='datetime64[ns]')
        )
        assert ds.TMP.isel(time=1, latitude=1, longitude=1).item() == 3


@pytest.mark.parametrize('group', ['', 'forecast/flx'])
def test_attribute_update_refreshes_consolidated_metadata(group: str):
    store = MemoryStore()
    write_batch(store, 'forecast/flx', field())
    set_group_attrs(store, group, description='original')

    set_group_attrs(store, group, description='updated')

    root = zarr.open_group(store, use_consolidated=True)
    target = root[group] if group else root
    assert target.attrs['description'] == 'updated'


def test_readiness_removal_refreshes_consolidated_metadata():
    store = MemoryStore()
    write_batch(store, 'forecast/flx', field())
    set_group_attrs(store, '', updated='2026-01-02')
    assert zarr.open_group(store, use_consolidated=True).attrs['updated'] == '2026-01-02'

    remove_group_attr(store, '', 'updated')

    assert 'updated' not in zarr.open_group(store, use_consolidated=True).attrs


def test_group_deletion_refreshes_consolidated_metadata_and_preserves_other_groups():
    store = MemoryStore()
    write_batch(store, 'forecast/flx', field())
    write_batch(store, 'reanalysis/pgb', field())
    assert 'forecast/flx' in zarr.open_group(store, use_consolidated=True)

    delete_group(store, 'forecast')

    root = zarr.open_group(store, use_consolidated=True)
    assert 'forecast' not in root
    assert 'reanalysis/pgb' in root
    with xr.open_zarr(store, group='reanalysis/pgb', consolidated=True, chunks=None) as ds:
        assert ds.TMP.isel(time=0, latitude=1, longitude=1).item() == 3


def test_group_recreation_replaces_consolidated_dates():
    store = MemoryStore()
    write_batch(store, 'forecast/flx', field())
    delete_group(store, 'forecast')

    write_batch(store, 'forecast/flx', field(day='2026-01-03'))

    with xr.open_zarr(store, group='forecast/flx', consolidated=True, chunks=None) as ds:
        assert ds.sizes['time'] == 1
        assert ds.time.to_numpy()[0] == np.datetime64('2026-01-03')


def test_point_selection_preserves_cmip6_latitude_orientation():
    ds = field(latitude=(-59.875, -25.125, 55.125, 89.875))
    assert select_point(ds, latitude=55.125, longitude=0.125).TMP.item() == 4
    descending = field(latitude=(89.875, 55.125, -25.125, -59.875))
    assert select_point(descending, latitude=55.125, longitude=0.125).TMP.item() == 2


@pytest.mark.parametrize('longitude', [-180, 180])
def test_global_longitude_seam(longitude: float):
    ds = field(longitude=(0, 90, 180, 270))
    assert select_point(ds, latitude=55.125, longitude=longitude).TMP.item() == 2
    assert select_point(ds, latitude=55.125, longitude=-1).TMP.item() == 0


def test_regional_dateline_and_bounds():
    ds = field(longitude=(170, 175, -180, -175))
    np.testing.assert_allclose(ds.longitude, [170, 175, 180, 185])
    assert select_point(ds, latitude=55.125, longitude=-175).TMP.item() == 3
    assert select_point(ds, latitude=55.125, longitude=167.5).TMP.item() == 0
    assert select_point(ds, latitude=55.125, longitude=-172.5).TMP.item() == 3
    for lon in (160, -170, 0):
        with pytest.raises(StorageError, match='outside'):
            select_point(ds, latitude=55.125, longitude=lon)
    for lat in (54.99, 55.51):
        with pytest.raises(StorageError, match='outside'):
            select_point(ds, latitude=lat, longitude=175)


def test_nearest_ties_choose_larger_coordinate():
    ds = field()
    selected = select_point(ds, latitude=55.25, longitude=0.25)
    assert selected.latitude.item() == 55.375
    assert selected.longitude.item() == 0.375
    assert selected.TMP.item() == 3


def test_separate_grid_groups_reject_conflicting_append():
    store = MemoryStore()
    flx = field(latitude=(54.7, 55.65), longitude=(0, 0.9375))
    pgb = field(latitude=(54, 55), longitude=(0, 1))
    write_batch(store, 'forecast/flx', flx)
    write_batch(store, 'forecast/pgb', pgb)
    np.testing.assert_array_equal(open_dataset(store, 'forecast/flx').latitude, flx.latitude)
    np.testing.assert_array_equal(open_dataset(store, 'forecast/pgb').latitude, pgb.latitude)
    with pytest.raises(StorageError, match='conflicting'):
        write_batch(store, 'forecast/flx', pgb)


def test_append_and_conflict_leave_committed_data_unchanged():
    store = MemoryStore()
    write_batch(store, 'reanalysis/flx', field())
    write_batch(store, 'reanalysis/flx', field(day='2026-01-02'))
    actual = open_dataset(store, 'reanalysis/flx')
    committed = actual.load().copy(deep=True)
    np.testing.assert_array_equal(
        actual.time.to_numpy(), np.array(['2026-01-01', '2026-01-02'], dtype='datetime64[ns]')
    )
    assert CRS.from_wkt(actual.spatial_ref.attrs['crs_wkt']).to_epsg() == 4326
    with pytest.raises(StorageError, match='follow'):
        write_batch(store, 'reanalysis/flx', field(day='2026-01-02'))
    xr.testing.assert_identical(open_dataset(store, 'reanalysis/flx'), committed)
    changed = field(day='2026-01-03')
    changed.TMP.attrs['units'] = 'degC'
    with pytest.raises(StorageError, match='conflicting'):
        write_batch(store, 'reanalysis/flx', changed)
    xr.testing.assert_identical(open_dataset(store, 'reanalysis/flx'), committed)


def test_failed_append_is_not_published_and_retry_recovers_tail(monkeypatch: pytest.MonkeyPatch):
    store = MemoryStore()
    write_batch(store, 'grid', field())
    real_write = xr.Dataset.to_zarr

    def fail_after_partial_write(_self, *_args, **_kwargs):  # noqa: ANN001
        target = zarr.open_group(store).get_group('grid')
        array = target.get_array('TMP')
        array.resize((2, 2, 2))
        array[1] = 999
        msg = 'Simulated interrupted upload'
        raise OSError(msg)

    monkeypatch.setattr(xr.Dataset, 'to_zarr', fail_after_partial_write)
    with pytest.raises(OSError, match='interrupted'):
        write_batch(store, 'grid', field(day='2026-01-02'))
    with pytest.raises(StorageError, match='completed'):
        open_dataset(store, 'grid')
    assert zarr.open_group(store).metadata.consolidated_metadata is None
    assert zarr.open_group(store)['grid'].attrs['write_state'] == 'writing'
    assert zarr.open_group(store)['grid'].attrs['committed_length'] == 1
    monkeypatch.setattr(xr.Dataset, 'to_zarr', real_write)
    write_batch(store, 'grid', field(day='2026-01-02'))
    recovered = open_dataset(store, 'grid')
    np.testing.assert_array_equal(recovered.TMP[0], field().TMP[0])
    np.testing.assert_array_equal(recovered.TMP[1], field().TMP[0])
    with xr.open_zarr(store, group='grid', consolidated=True, chunks=None) as ds:
        assert ds.sizes['time'] == ds.attrs['committed_length'] == 2
        assert ds.attrs['write_state'] == 'complete'


def test_streamed_month_is_published_only_after_iterator_finishes_and_retry_recovers():
    from weatheasy.storage import last_date

    store = MemoryStore()
    write_batch(store, 'grid', field())
    before = open_dataset(store, 'grid').load()

    def broken_batches():
        yield field(day='2026-01-02')
        assert zarr.open_group(store)['grid'].attrs['committed_length'] == 1
        with pytest.raises(StorageError, match='completed'):
            open_dataset(store, 'grid')
        # Represents failure while decoding the next day or publishing its cache.
        msg = 'interrupted before the month is complete'
        raise OSError(msg)

    with pytest.raises(OSError, match='interrupted'):
        write_batches(store, 'grid', broken_batches())
    assert str(last_date(store, 'grid')) == '2026-01-01'
    write_batches(store, 'grid', (field(day=f'2026-01-0{day}') for day in (2, 3)))
    with open_dataset(store, 'grid') as result:
        assert result.sizes['time'] == 3
        np.testing.assert_array_equal(result.TMP[0], before.TMP[0])
        np.testing.assert_array_equal(result.TMP[1], field().TMP[0])
        np.testing.assert_array_equal(result.TMP[2], field().TMP[0])


def test_zarr_read_does_not_import_grib_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / 'prepared.zarr'
    store = get_store(path)
    write_batch(store, 'grid', field())
    original = builtins.__import__

    def reject_decoder(name: str, *args, **kwargs):
        if name.split('.', maxsplit=1)[0] in {'cfgrib', 'eccodes'}:
            msg = f'Unexpected decoder import: {name}'
            raise AssertionError(msg)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_decoder)
    assert open_dataset(store, 'grid').TMP.sel(latitude=55.375, longitude=0.375).item() == 3


def test_local_uri_needs_no_s3_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original = builtins.__import__

    def reject_s3fs(name: str, *args, **kwargs):
        if name == 's3fs':
            msg = 'Local storage must not import s3fs'
            raise AssertionError(msg)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_s3fs)
    path = tmp_path / 'prepared.zarr'
    local = get_store(path)
    from_string = get_store(str(path))
    assert isinstance(local, LocalStore)
    assert isinstance(from_string, LocalStore)
    assert local.root == from_string.root == path


def test_s3_uri_reuses_supplied_filesystem_without_network():
    s3fs = pytest.importorskip('s3fs')
    filesystem = s3fs.S3FileSystem(asynchronous=True)
    weather = get_store('s3://weather-bucket/zarr', filesystem=filesystem)
    dem = get_store('s3://weather-bucket/dem.zarr', filesystem=filesystem)
    assert isinstance(weather, FsspecStore)
    assert isinstance(dem, FsspecStore)
    assert weather.fs is dem.fs is filesystem
    assert weather.path == 'weather-bucket/zarr'
    assert dem.path == 'weather-bucket/dem.zarr'


def test_open_dataset_resolves_s3_uri_for_metadata_and_xarray(
    monkeypatch: pytest.MonkeyPatch,
):
    s3fs = pytest.importorskip('s3fs')
    filesystem = s3fs.S3FileSystem(asynchronous=True)
    seen = []

    def get_group(group: str) -> SimpleNamespace:
        assert group == 'daily'
        return SimpleNamespace(
            attrs={
                'storage_schema': 'weatheasy-coordinate-v1',
                'write_state': 'complete',
                'committed_length': 1,
            }
        )

    def open_group(
        store: StoreLike, *, mode: str, zarr_format: int, use_consolidated: bool
    ) -> SimpleNamespace:
        assert mode == 'r'
        assert zarr_format == 3
        assert use_consolidated is False
        seen.append(store)
        return SimpleNamespace(get_group=get_group)

    def open_zarr(store: StoreLike, *, group: str, **kwargs):
        assert group == 'daily'
        assert kwargs['decode_coords'] == 'all'
        seen.append(store)
        return field()

    monkeypatch.setattr(zarr, 'open_group', open_group)
    monkeypatch.setattr(xr, 'open_zarr', open_zarr)
    store = get_store('s3://weather-bucket/zarr', filesystem=filesystem)
    result = open_dataset(store, 'daily')
    assert result.TMP.sel(latitude=55.375, longitude=0.375).item() == 3
    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert isinstance(seen[0], FsspecStore)
    assert seen[0].fs is filesystem
    assert seen[0].path == 'weather-bucket/zarr'


def test_get_storage_preserves_shared_s3_filesystem(monkeypatch: pytest.MonkeyPatch):
    s3fs = pytest.importorskip('s3fs')
    filesystem = s3fs.S3FileSystem(asynchronous=True)

    def open_group(
        store: StoreLike, *, mode: str, zarr_format: int, use_consolidated: bool
    ) -> StoreLike:
        assert isinstance(store, FsspecStore)
        assert store.fs is filesystem
        assert store.path == 'weather-bucket/zarr'
        assert mode == 'a'
        assert zarr_format == 3
        assert use_consolidated is False
        return store

    monkeypatch.setattr(zarr, 'open_group', open_group)
    result = get_storage('s3://weather-bucket/zarr', filesystem=filesystem)
    assert isinstance(result, FsspecStore)
    assert result.fs is filesystem


def test_store_rejects_missing_s3_bucket_or_filesystem_for_local_path(tmp_path: Path):
    with pytest.raises(StorageError, match='bucket'):
        get_store('s3://')
    with pytest.raises(StorageError, match='bucket'):
        get_store('s3:///without-bucket')
    s3fs = pytest.importorskip('s3fs')
    with pytest.raises(StorageError, match='only be used'):
        get_store(tmp_path / 'data', filesystem=s3fs.S3FileSystem(asynchronous=True))
