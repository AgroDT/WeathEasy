from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from weatheasy import const
from weatheasy.sources import read_grib, read_netcdf
from weatheasy.storage import open_dataset, select_point, write_batch


TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


def grib_fixture(path: Path, *, gaussian: bool = False, spacing: int = 1):
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
            updates, dataDate=20260922, dataTime=0, shapeOfTheEarth=8, shortName='2t'
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
        ('forecast/flx', True, 1),
        ('forecast/pgb', False, 1),
        ('reanalysis/flx', False, 0.5),
        ('reanalysis/pgb', False, 0.5),
    ],
)
def test_grib_source_to_zarr_uses_product_metadata_and_actual_grid(
    tmp_path: Path,
    *,
    group: str,
    gaussian: bool,
    spacing: int,
):
    path = tmp_path / 'source.grb2'
    latitudes, values = grib_fixture(path, gaussian=gaussian, spacing=spacing)
    ds = read_grib(
        path, name='TMP', filter_by_keys={'shortName': '2t'}, crs=const.CFS2_CRS, units='K'
    )
    np.testing.assert_allclose(ds.latitude, latitudes[::-1], atol=1e-8)
    np.testing.assert_allclose(ds.TMP[0], values[::-1], atol=1e-4)
    assert ds.TMP.attrs['units'] == 'K'  # No implicit GDAL Celsius conversion.
    assert ds.time.item() == np.datetime64('2026-09-22', 'ns').item()
    assert not list(tmp_path.glob('*.idx'))
    assert ds.attrs['source_grib']['TMP']['radius'] == 6371200
    store = MemoryStore()
    write_batch(store, group, ds)
    with open_dataset(store, group) as result:
        assert result.sizes['latitude'] == (190 if gaussian else 3)
        point = select_point(result, latitude=-30.4 if gaussian else 55, longitude=0)
        row = int(np.argmin(abs(latitudes - (-30.4 if gaussian else 55))))
        np.testing.assert_allclose(point.TMP.item(), values[row, 0], atol=1e-4)
        assert result.rio.crs.to_dict()['R'] == 6371229
        assert result.attrs['source_grib'] == ds.attrs['source_grib']
        np.testing.assert_array_equal(result.latitude, ds.latitude)
        np.testing.assert_array_equal(result.longitude, ds.longitude)
        np.testing.assert_array_equal(result.TMP, ds.TMP)
        assert 'GeoTransform' not in result.spatial_ref.attrs
    # Gaussian axes remain nonuniform, not a synthetic dy extracted from bbox.
    if gaussian:
        assert np.ptp(np.diff(ds.latitude.to_numpy())) > 0.001


@pytest.mark.parametrize('source_attrs', [{}, {'units': 'incorrect'}])
def test_netcdf_cmip6_actual_ascending_latitude(tmp_path: Path, source_attrs: dict):
    path = tmp_path / 'cmip6.nc'
    original = xr.Dataset(
        {
            'tas': (
                ('time', 'lat', 'lon'),
                np.arange(12, dtype='float32').reshape(1, 4, 3),
                source_attrs,
            )
        },
        coords={
            'time': [np.datetime64('2000-01-01')],
            'lat': [-59.875, -25.125, 55.125, 89.875],
            'lon': [0.125, 0.375, 0.625],
        },
    )
    original.to_netcdf(path, engine='netcdf4')
    ds = read_netcdf(path, units={'tas': 'K'}, crs=const.CMIP6_CRS)
    store = MemoryStore()
    write_batch(store, 'ACCESS-CM2/historical/r1i1p1f1', ds)
    result = open_dataset(store, 'ACCESS-CM2/historical/r1i1p1f1')
    assert select_point(result, latitude=55.125, longitude=0.375).tas.item() == 7
    assert result.tas.attrs['units'] == 'K'
    assert result.rio.crs.to_epsg() == 4326
    assert result.attrs['source_units'] == {'tas': source_attrs.get('units')}
    np.testing.assert_array_equal(result.tas, original.tas)


def test_pevpr_selects_instant_and_converts_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from weatheasy import const
    from weatheasy.sources import read_cfs2

    ec = pytest.importorskip('eccodes')
    path = tmp_path / 'pevpr.grb2'
    grib_fixture(path, gaussian=False, spacing=1)
    with path.open('rb') as stream:
        template = ec.codes_grib_new_from_file(stream)
    with path.open('wb') as stream:
        for statistic, value in [('avg', 900), ('instant', 100)]:
            handle = ec.codes_clone(template)
            for key, val in {
                'centre': 'kwbc',
                'discipline': 0,
                'parameterCategory': 1,
                'parameterNumber': 200,
                'typeOfLevel': 'surface',
                'level': 0,
                'stepType': statistic,
                'startStep': 0,
                'endStep': 0,
            }.items():
                ec.codes_set(handle, key, val)
            ec.codes_set_values(handle, np.full(12, value))
            ec.codes_write(handle, stream)
            ec.codes_release(handle)
    ec.codes_release(template)
    monkeypatch.setattr(const, 'CFS2_FLX_BANDS', {'PEVPR': const.CFS2_FLX_BANDS['PEVPR']})
    result = read_cfs2(path, kind='flx')
    np.testing.assert_allclose(result.PEVPR, 3.526530612, rtol=1e-6)
    assert result.PEVPR.attrs['units'] == 'mm/day'


def test_cfs2_shared_indexes_preserve_fields_layers_and_file_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from weatheasy.sources import read_cfs2

    ec = pytest.importorskip('eccodes')
    names = ('TMP', 'TMIN', 'SOILW_0-0.1m', 'SOILW_0.1-0.4m')
    monkeypatch.setattr(
        const, 'CFS2_FLX_BANDS', {name: const.CFS2_FLX_BANDS[name] for name in names}
    )
    for index in range(2):
        path = tmp_path / f'fields-{index}.grb2'
        grib_fixture(path)
        with path.open('rb') as stream:
            template = ec.codes_grib_new_from_file(stream)
        try:
            with path.open('wb') as stream:
                for discipline, category, number, top, bottom, value in (
                    (0, 0, 0, None, None, 290 + index),
                    (0, 0, 5, None, None, 270 + index),
                    (2, 0, 192, 0, 10, 0.1 + index),
                    (2, 0, 192, 10, 40, 0.3 + index),
                ):
                    handle = ec.codes_clone(template)
                    try:
                        updates = {
                            'centre': 'kwbc',
                            'discipline': discipline,
                            'parameterCategory': category,
                            'parameterNumber': number,
                        }
                        if top is not None:
                            updates.update(
                                typeOfLevel='depthBelowLandLayer',
                                scaleFactorOfFirstFixedSurface=2,
                                scaledValueOfFirstFixedSurface=top,
                                scaleFactorOfSecondFixedSurface=2,
                                scaledValueOfSecondFixedSurface=bottom,
                            )
                        for key, val in updates.items():
                            ec.codes_set(handle, key, val)
                        ec.codes_set_values(handle, np.full(12, value))
                        ec.codes_write(handle, stream)
                    finally:
                        ec.codes_release(handle)
        finally:
            ec.codes_release(template)
        result = read_cfs2(path, kind='flx')
        for name, expected in zip(
            names, (16.85 + index, -3.15 + index, 0.1 + index, 0.3 + index), strict=True
        ):
            np.testing.assert_allclose(result[name], expected, atol=1e-4)
        assert not list(tmp_path.glob('*.idx'))
