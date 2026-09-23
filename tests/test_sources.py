from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from weatheasy.sources import read_grib, read_netcdf
from weatheasy.storage import open_dataset, select_point, write_batch


def grib_fixture(path, *, gaussian=False, spacing=1):
    ec = pytest.importorskip('eccodes')
    if gaussian:
        handle = ec.codes_grib_new_from_samples('regular_gg_sfc_grib2')
        latitudes = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(190)[0]))[::-1]
        width, height = 384, 190
        updates = {
            'N': 95,
            'Ni': width,
            'Nj': height,
            'latitudeOfFirstGridPointInDegrees': float(latitudes[0]),
            'latitudeOfLastGridPointInDegrees': float(latitudes[-1]),
            'longitudeOfFirstGridPointInDegrees': 0,
            'longitudeOfLastGridPointInDegrees': 359.0625,
            'iDirectionIncrementInDegrees': 0.9375,
        }
    else:
        handle = ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
        width, height = 4, 3
        latitudes = 56 - np.arange(height) * spacing
        updates = {
            'Ni': width,
            'Nj': height,
            'latitudeOfFirstGridPointInDegrees': 56,
            'latitudeOfLastGridPointInDegrees': float(latitudes[-1]),
            'longitudeOfFirstGridPointInDegrees': 0,
            'longitudeOfLastGridPointInDegrees': (width - 1) * spacing,
            'iDirectionIncrementInDegrees': spacing,
            'jDirectionIncrementInDegrees': spacing,
        }
    try:
        for key, value in dict(
            updates, dataDate=20260922, dataTime=0, shapeOfTheEarth=6, shortName='2t'
        ).items():
            ec.codes_set(handle, key, value)
        values = np.arange(width * height, dtype=np.float64).reshape(height, width) / 100 + 273.15
        ec.codes_set_values(handle, values.ravel())
        with path.open('wb') as stream:
            ec.codes_write(handle, stream)
    finally:
        ec.codes_release(handle)
    return latitudes, values


@pytest.mark.parametrize(
    ('group', 'gaussian', 'spacing'),
    [
        ('cfs2/forecast/flx', True, 1),
        ('cfs2/forecast/pgb', False, 1),
        ('cfs2/reanalysis/flx', False, 0.5),
        ('cfs2/reanalysis/pgb', False, 0.5),
    ],
)
def test_grib_source_to_zarr_preserves_grid_and_units(tmp_path, group, gaussian, spacing):
    path = tmp_path / 'source.grb2'
    latitudes, values = grib_fixture(path, gaussian=gaussian, spacing=spacing)
    ds = read_grib(path, name='TMP', filter_by_keys={'shortName': '2t'})
    np.testing.assert_allclose(ds.latitude, latitudes[::-1], atol=1e-8)
    np.testing.assert_allclose(ds.TMP[0], values[::-1], atol=1e-4)
    assert ds.TMP.attrs['units'] == 'K'  # No implicit GDAL Celsius conversion.
    assert ds.time.item() == np.datetime64('2026-09-22', 'ns').item()
    assert not list(tmp_path.glob('*.idx'))
    store = MemoryStore()
    write_batch(store, group, ds)
    with open_dataset(store, group) as result:
        assert result.sizes['latitude'] == (190 if gaussian else 3)
        point = select_point(result, latitude=-30.4 if gaussian else 55, longitude=0)
        row = int(np.argmin(abs(latitudes - (-30.4 if gaussian else 55))))
        np.testing.assert_allclose(point.TMP.item(), values[row, 0], atol=1e-4)
        assert result.spatial_ref.attrs == ds.spatial_ref.attrs
        assert result.rio.crs == ds.rio.crs
        assert result.spatial_ref.attrs['grid_mapping_name'] == 'latitude_longitude'
        assert result.spatial_ref.attrs['semi_major_axis'] == 6371229
        assert result.spatial_ref.attrs['semi_minor_axis'] == 6371229
        assert result.latitude.attrs['units'] == 'degrees_north'
        assert result.longitude.attrs['axis'] == 'X'
        np.testing.assert_array_equal(result.latitude, ds.latitude)
        np.testing.assert_array_equal(result.longitude, ds.longitude)
        np.testing.assert_array_equal(result.TMP, ds.TMP)
        assert 'GeoTransform' not in result.spatial_ref.attrs
    # Gaussian axes remain nonuniform, not a synthetic dy extracted from bbox.
    if gaussian:
        assert np.ptp(np.diff(ds.latitude.to_numpy())) > 0.001


def test_netcdf_cmip6_actual_ascending_latitude(tmp_path):
    path = tmp_path / 'cmip6.nc'
    original = xr.Dataset(
        {
            'tas': (
                ('time', 'lat', 'lon'),
                np.arange(12, dtype='float32').reshape(1, 4, 3),
                {'units': 'K'},
            )
        },
        coords={
            'time': [np.datetime64('2000-01-01')],
            'lat': [-59.875, -25.125, 55.125, 89.875],
            'lon': [0.125, 0.375, 0.625],
        },
    )
    original.to_netcdf(path, engine='netcdf4')
    ds = read_netcdf(path, variables=['tas'], crs='EPSG:4326')
    store = MemoryStore()
    write_batch(store, 'cmip6/ACCESS-CM2/historical/r1i1p1f1', ds)
    result = open_dataset(store, 'cmip6/ACCESS-CM2/historical/r1i1p1f1')
    assert select_point(result, latitude=55.125, longitude=0.375).tas.item() == 7
    np.testing.assert_array_equal(result.tas, original.tas)
