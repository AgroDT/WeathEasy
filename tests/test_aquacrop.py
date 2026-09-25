"""FAO published examples and prepared Zarr -> AquaCrop integration."""

from datetime import date

import numpy as np
import pytest
import xarray as xr
import zarr

from weatheasy.aquacrop_adapter import (
    et0_fao56,
    extraterrestrial_radiation,
    get_weather_data,
    net_longwave_radiation,
)
from weatheasy.storage import prepare_dataset, write_batch
from weatheasy.util import get_storage


# FAO-56 chapter 4, example 18, Uccle, July 6:
# https://www.fao.org/4/x0490e/x0490e08.htm
# Published ea/es are translated to equivalent RHmean (not RHmin/RHmax mean).
UCCLE = {
    't_min': 12.3,
    't_max': 21.5,
    't_mean': 16.9,
    'rh_mean': 100 * 1.409 / 1.997,
    'wind': 2.78,
    'pressure': 100.1,
    'rs': 22.07,
    'rs0': 30.90,
}


def test_published_fao_examples():
    # Chapter 3 examples 8 and 11 (rounded published results):
    # https://www.fao.org/4/x0490e/x0490e07.htm
    assert extraterrestrial_radiation(-20, 246) == pytest.approx(32.2, abs=0.05)
    assert net_longwave_radiation(19.1, 25.1, 2.1, 14.5, 18.8) == pytest.approx(3.5, abs=0.05)
    assert et0_fao56(**UCCLE) == pytest.approx(3.88, abs=0.02)
    assert extraterrestrial_radiation(50.8, 187) == pytest.approx(41.09, abs=0.02)


def test_arrays_mean_wind_height_and_nan():
    scalar = et0_fao56(**UCCLE)
    result = et0_fao56(**(UCCLE | {'t_mean': [16.9, 23, np.nan]}))
    assert result.shape == (3,)
    assert result[0] == scalar
    assert result[1] != scalar
    assert np.isnan(result[2])
    # FAO example 18 explicitly gives 2.078 m/s at 2 m from 2.78 m/s at 10 m.
    assert et0_fao56(**(UCCLE | {'wind': 2.078}), wind_height=2) == pytest.approx(scalar, abs=0.002)
    for key in UCCLE:
        assert np.isnan(et0_fao56(**(UCCLE | {key: np.nan})))


def test_cloud_factor_and_polar_radiation():
    clear = net_longwave_radiation(19.1, 25.1, 2.1, 18.8, 18.8)
    overcast = net_longwave_radiation(19.1, 25.1, 2.1, 0, 18.8)
    assert overcast == pytest.approx(clear * 0.05)
    assert net_longwave_radiation(19.1, 25.1, 2.1, 100, 18.8) == clear
    np.testing.assert_allclose(extraterrestrial_radiation(90, [1, 355]), 0)
    assert extraterrestrial_radiation(90, 172) > 0
    assert extraterrestrial_radiation(-90, 172) == 0
    assert np.isfinite(et0_fao56(**(UCCLE | {'rs0': 0, 'rs': 0})))
    assert np.isfinite(et0_fao56(**(UCCLE | {'rs0': 1e-15})))


def weather_store(tmp_path, correction=()):
    root = get_storage(str(tmp_path / 'weather'))
    # Includes leap day and a month boundary; only WIND is provided, no U/V.
    dates = np.arange('2020-02-28', '2020-03-03', dtype='datetime64[D]')
    values = {
        'TMIN': 12.3,
        'TMAX': 21.5,
        'TMP': 16.9,
        'WIND': 2.78,
        'PRES': 1001,
        'RH': 100 * 1.409 / 1.997,
        'DSWRF': 22.07 / 0.0864,
        'PRATE': np.array([8, np.nan, 0.5, 2]) / 86400,
    }
    coords = {'time': dates, 'latitude': [50.8, 50.9], 'longitude': [4.3, 4.4]}

    units = {
        'TMIN': 'degC',
        'TMAX': 'degC',
        'TMP': 'degC',
        'WIND': 'm s-1',
        'PRES': 'hPa',
        'RH': '%',
        'DSWRF': 'W m-2',
        'PRATE': 'kg m-2 s-1',
        'TMP_corr': 'degC',
        'PRATE_corr': 'mm day-1',
    }

    def dataset(fields):
        return prepare_dataset(
            xr.Dataset(
                {
                    name: (
                        ('time', 'latitude', 'longitude'),
                        np.broadcast_to(
                            np.broadcast_to(value, (4,))[:, None, None], (4, 2, 2)
                        ).astype(np.float64),
                        {'units': units[name]},
                    )
                    for name, value in fields.items()
                },
                coords=coords,
            ),
            crs='EPSG:4326',
        )

    write_batch(root.store, 'era5/daily', dataset(values))
    corrected = {'TMP_corr': [23, np.nan, 23, 23], 'PRATE_corr': [12, np.nan, 0.25, 1]}
    if correction:
        write_batch(
            root.store, 'era5/correction', dataset({key: corrected[key] for key in correction})
        )
    return root


def read_weather(root, **kwargs):
    return get_weather_data(
        root=root,
        latitude=50.8,
        longitude=4.3,
        altitude=100,
        start=date(2020, 2, 28),
        end=date(2020, 3, 2),
        **kwargs,
    )


@pytest.mark.parametrize(
    'correction', [(), ('TMP_corr',), ('PRATE_corr',), ('TMP_corr', 'PRATE_corr')]
)
def test_adapter_units_independent_fallback_and_readonly(tmp_path, correction):
    root = weather_store(tmp_path, correction)
    root = zarr.open_group(root.store, mode='r')
    raw = read_weather(root, use_corrected=False)
    actual = read_weather(root)
    np.testing.assert_allclose(raw.pr, [8, np.nan, 0, 2], rtol=1e-6)
    np.testing.assert_allclose(
        actual.pr, [12, np.nan, 0, 1] if 'PRATE_corr' in correction else raw.pr
    )
    np.testing.assert_allclose(actual.tmin, 12.3)
    np.testing.assert_allclose(actual.tmax, 21.5)
    assert actual.dtype.names == ('tmin', 'tmax', 'pr', 'et0')
    if 'TMP_corr' in correction:
        corrected_et0 = et0_fao56(
            **(UCCLE | {'t_mean': 23, 'rs0': 0.752 * extraterrestrial_radiation(50.8, 59)})
        )
        assert actual.et0[0] == pytest.approx(corrected_et0)
        assert np.isnan(actual.et0[1])  # Existing corrected NaN never falls back to raw.
    else:
        np.testing.assert_allclose(actual.et0, raw.et0)
    # Verify pressure/radiation conversions and leap/month day ordering through real getter.
    expected = et0_fao56(
        **(
            UCCLE
            | {
                'rs0': 0.752 * extraterrestrial_radiation(50.8, [59, 60, 61, 62]),
            }
        )
    )
    np.testing.assert_allclose(raw.et0, np.where(expected > 0, expected, np.nan), rtol=1e-6)


def test_corrected_precip_missing_day_and_nonpositive_et0(tmp_path):
    root = weather_store(tmp_path, ('PRATE_corr',))
    root['era5/correction/PRATE_corr'][3] = np.nan
    # Zero radiation and saturated air give negative net radiation and ET0.
    root['era5/daily/DSWRF'][3] = 0
    root['era5/daily/RH'][3] = 100
    actual = read_weather(root)
    assert np.isnan(actual.pr[3])
    assert np.isnan(actual.et0[3])
    assert read_weather(root, use_corrected=False).pr[3] == pytest.approx(2)
