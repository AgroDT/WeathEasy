from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
import zarr
from rasterio.crs import CRS
from zarr.storage import MemoryStore

from weatheasy.storage import (
    StorageError,
    merge_fields,
    open_dataset,
    prepare_dataset,
    select_point,
    write_batch,
)


def field(*, latitude=(55.125, 55.375), longitude=(0.125, 0.375), day='2026-01-01'):
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


def test_round_trip_metadata_nan_and_direct_xarray(tmp_path):
    ds = field()
    ds.TMP.data[0, 0, 0] = np.nan
    path = tmp_path / 'weather.zarr'
    write_batch(path, 'cmip6/ACCESS-CM2/historical/r1i1p1f1', ds, tile_size=2)
    with xr.open_zarr(
        path,
        group='cmip6/ACCESS-CM2/historical/r1i1p1f1',
        consolidated=False,
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
    arr = root['cmip6/ACCESS-CM2/historical/r1i1p1f1/TMP']
    assert arr.metadata.zarr_format == 3
    assert arr.metadata.dimension_names == ('time', 'latitude', 'longitude')
    assert arr.chunks == (1, 2, 2)
    # A new process importing only xarray can read the artifact.
    script = """import xarray as xr, sys
x=xr.open_zarr(sys.argv[1],group='cmip6/ACCESS-CM2/historical/r1i1p1f1',chunks=None,consolidated=False,decode_coords='all')
assert x.TMP.sel(latitude=55.375,longitude=0.375).item()==3
assert 'weatheasy' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script, str(path)], check=True)  # noqa: S603


def test_point_selection_preserves_cmip6_latitude_orientation():
    ds = field(latitude=(-59.875, -25.125, 55.125, 89.875))
    assert select_point(ds, latitude=55.125, longitude=0.125).TMP.item() == 4
    descending = field(latitude=(89.875, 55.125, -25.125, -59.875))
    assert select_point(descending, latitude=55.125, longitude=0.125).TMP.item() == 2


@pytest.mark.parametrize('longitude', [-180, 180])
def test_global_longitude_seam(longitude):
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


def test_separate_grid_groups_and_no_implicit_alignment():
    store = MemoryStore()
    flx = field(latitude=(54.7, 55.65), longitude=(0, 0.9375))
    pgb = field(latitude=(54, 55), longitude=(0, 1))
    write_batch(store, 'cfs2/forecast/flx', flx)
    write_batch(store, 'cfs2/forecast/pgb', pgb)
    np.testing.assert_array_equal(open_dataset(store, 'cfs2/forecast/flx').latitude, flx.latitude)
    np.testing.assert_array_equal(open_dataset(store, 'cfs2/forecast/pgb').latitude, pgb.latitude)
    with pytest.raises(ValueError, match='align'):
        merge_fields(flx, pgb.rename(TMP='PRES'))
    with pytest.raises(StorageError, match='conflicting'):
        write_batch(store, 'cfs2/forecast/flx', pgb)


def test_append_and_conflict_leave_committed_data_unchanged():
    store = MemoryStore()
    write_batch(store, 'cfs2/reanalysis/flx', field())
    write_batch(store, 'cfs2/reanalysis/flx', field(day='2026-01-02'))
    actual = open_dataset(store, 'cfs2/reanalysis/flx')
    np.testing.assert_array_equal(
        actual.time.to_numpy(), np.array(['2026-01-01', '2026-01-02'], dtype='datetime64[ns]')
    )
    assert CRS.from_wkt(actual.spatial_ref.attrs['crs_wkt']).to_epsg() == 4326
    with pytest.raises(StorageError, match='follow'):
        write_batch(store, 'cfs2/reanalysis/flx', field(day='2026-01-02'))
    changed = field(day='2026-01-03')
    changed.TMP.attrs['units'] = 'degC'
    with pytest.raises(StorageError, match='conflicting'):
        write_batch(store, 'cfs2/reanalysis/flx', changed)
    assert open_dataset(store, 'cfs2/reanalysis/flx').sizes['time'] == 2


def test_failed_append_is_not_published_and_retry_recovers_tail(monkeypatch):
    store = MemoryStore()
    write_batch(store, 'grid', field())
    real_write = xr.Dataset.to_zarr

    def fail_after_partial_write(_self, *_args, **_kwargs):
        target = zarr.open_group(store)['grid']
        target['TMP'].resize((2, 2, 2))
        target['TMP'][1] = 999
        msg = 'Simulated interrupted upload'
        raise OSError(msg)

    monkeypatch.setattr(xr.Dataset, 'to_zarr', fail_after_partial_write)
    with pytest.raises(OSError, match='interrupted'):
        write_batch(store, 'grid', field(day='2026-01-02'))
    with pytest.raises(StorageError, match='completed'):
        open_dataset(store, 'grid')
    assert zarr.open_group(store)['grid'].attrs['committed_length'] == 1
    monkeypatch.setattr(xr.Dataset, 'to_zarr', real_write)
    write_batch(store, 'grid', field(day='2026-01-02'))
    recovered = open_dataset(store, 'grid')
    np.testing.assert_array_equal(recovered.TMP[0], field().TMP[0])
    np.testing.assert_array_equal(recovered.TMP[1], field().TMP[0])


def test_zarr_read_does_not_import_grib_decoder(tmp_path, monkeypatch):
    path = tmp_path / 'prepared.zarr'
    write_batch(path, 'grid', field())
    original = builtins.__import__

    def reject_decoder(name, *args, **kwargs):
        if name.split('.')[0] in {'cfgrib', 'eccodes'}:
            msg = f'Unexpected decoder import: {name}'
            raise AssertionError(msg)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_decoder)
    assert open_dataset(path, 'grid').TMP.sel(latitude=55.375, longitude=0.375).item() == 3
