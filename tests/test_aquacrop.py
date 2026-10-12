"""FAO published examples and prepared Zarr -> AquaCrop integration."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from numpy.typing import ArrayLike

from weatheasy.aquacrop_adapter import (
    et0_fao56,
    extraterrestrial_radiation,
    get_weather_data,
    net_longwave_radiation,
)
from weatheasy.storage import StorageError, get_store, prepare_dataset, write_batch
from weatheasy.util import get_storage


@pytest.fixture(name='corrected_weather')
def create_corrected_weather(tmp_path: Path):
    return weather_store(tmp_path, ('PRATE_corr',))


def _et0_fao56_for_uccle(
    *,
    t_min: ArrayLike = 12.3,
    t_max: ArrayLike = 21.5,
    t_mean: ArrayLike = 16.9,
    rh_mean: ArrayLike = 100 * 1.409 / 1.997,
    wind: ArrayLike = 2.78,
    pressure: ArrayLike = 100.1,
    rs: ArrayLike = 22.07,
    rs0: ArrayLike = 30.90,
    wind_height: float = 10,
):
    """FAO-56 chapter 4, example 18, Uccle, July 6: https://www.fao.org/4/x0490e/x0490e08.htm.

    Published ea/es are translated to equivalent RHmean (not RHmin/RHmax mean).
    """

    return et0_fao56(
        t_min=t_min,
        t_max=t_max,
        t_mean=t_mean,
        rh_mean=rh_mean,
        wind=wind,
        pressure=pressure,
        rs=rs,
        rs0=rs0,
        wind_height=wind_height,
    )


@pytest.mark.parametrize(
    ('latitude', 'day', 'expected', 'tolerance'),
    [(-20, 246, 32.2, 0.05), (50.8, 187, 41.09, 0.02)],
)
def test_extraterrestrial_radiation_fao_examples(
    latitude: float, day: int, expected: float, tolerance: float
):
    # FAO-56 chapter 3, examples 8 and 11: https://www.fao.org/4/x0490e/x0490e07.htm.
    assert extraterrestrial_radiation(latitude, day) == pytest.approx(expected, abs=tolerance)


def test_longwave_radiation_fao_example():
    assert net_longwave_radiation(19.1, 25.1, 2.1, 14.5, 18.8) == pytest.approx(3.5, abs=0.05)


def test_et0_fao_uccle_example():
    assert _et0_fao56_for_uccle() == pytest.approx(3.88, abs=0.02)


def test_et0_array_matches_scalar():
    result = _et0_fao56_for_uccle(t_mean=[16.9, 16.9])
    assert result.shape == (2,)
    np.testing.assert_allclose(result, _et0_fao56_for_uccle())


def test_et0_uses_mean_temperature():
    result = _et0_fao56_for_uccle(t_mean=[16.9, 23])
    assert result[1] != result[0]


def test_et0_preserves_missing_temperature():
    result = _et0_fao56_for_uccle(t_mean=[16.9, np.nan])
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_et0_converts_wind_height():
    # FAO example 18 gives 2.078 m/s at 2 m from 2.78 m/s at 10 m.
    assert _et0_fao56_for_uccle(wind=2.078, wind_height=2) == pytest.approx(
        _et0_fao56_for_uccle(), abs=0.002
    )


def test_overcast_longwave_radiation():
    clear = net_longwave_radiation(19.1, 25.1, 2.1, 18.8, 18.8)
    assert net_longwave_radiation(19.1, 25.1, 2.1, 0, 18.8) == pytest.approx(clear * 0.05)


def test_longwave_radiation_caps_clear_sky_ratio():
    clear = net_longwave_radiation(19.1, 25.1, 2.1, 18.8, 18.8)
    assert net_longwave_radiation(19.1, 25.1, 2.1, 100, 18.8) == clear


@pytest.mark.parametrize(('latitude', 'day'), [(90, [1, 355]), (-90, 172)])
def test_polar_night_radiation(latitude: float, day: int | list[int]):
    np.testing.assert_allclose(extraterrestrial_radiation(latitude, day), 0)


def test_polar_day_radiation():
    assert extraterrestrial_radiation(90, 172) > 0


@pytest.mark.parametrize(('rs0', 'rs'), [(0, 0), (1e-15, 22.07)])
def test_et0_handles_negligible_clear_sky_radiation(rs0: float, rs: float):
    assert np.isfinite(_et0_fao56_for_uccle(rs0=rs0, rs=rs))


def weather_store(tmp_path: Path, correction: tuple[str, ...] = ()):
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

    def dataset(fields: dict):
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

    write_batch(root.store, 'daily', dataset(values))
    corrected = {'TMP_corr': [23, np.nan, 23, 23], 'PRATE_corr': [12, np.nan, 0.25, 1]}
    if correction:
        write_batch(root.store, 'correction', dataset({key: corrected[key] for key in correction}))
    return root


def read_weather(root: zarr.Group, **kwargs):
    return get_weather_data(
        store=root.store,
        latitude=50.8,
        longitude=4.3,
        altitude=100,
        start=date(2020, 2, 28),
        end=date(2020, 3, 2),
        **kwargs,
    )


@pytest.mark.parametrize(
    'correction',
    [(), ('TMP_corr',), ('PRATE_corr',), ('TMP_corr', 'PRATE_corr')],
    ids=['raw', 'temperature-only', 'precipitation-only', 'both-corrected'],
)
def test_adapter_units_independent_fallback_and_readonly(
    tmp_path: Path,
    correction: tuple[str, ...],
):
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
        corrected_et0 = _et0_fao56_for_uccle(
            t_mean=23,
            rs0=0.752 * extraterrestrial_radiation(50.8, 59),
        )
        assert actual.et0[0] == pytest.approx(corrected_et0)
        assert np.isnan(actual.et0[1])  # Existing corrected NaN never falls back to raw.
    else:
        np.testing.assert_allclose(actual.et0, raw.et0)
    # Verify pressure/radiation conversions and leap/month day ordering through real getter.
    expected = _et0_fao56_for_uccle(rs0=0.752 * extraterrestrial_radiation(50.8, [59, 60, 61, 62]))
    np.testing.assert_allclose(raw.et0, np.where(expected > 0, expected, np.nan), rtol=1e-6)


def test_corrected_precipitation_nan_does_not_fall_back(corrected_weather: zarr.Group):
    corrected_weather.get_array('correction/PRATE_corr')[3] = np.nan
    actual = read_weather(corrected_weather)
    assert np.isnan(actual.pr[3])
    assert read_weather(corrected_weather, use_corrected=False).pr[3] == pytest.approx(2)


def test_nonpositive_et0_becomes_nan(corrected_weather: zarr.Group):
    # Zero radiation and saturated air give negative net radiation and ET0.
    corrected_weather.get_array('daily/DSWRF')[3] = 0
    corrected_weather.get_array('daily/RH')[3] = 100
    assert np.isnan(read_weather(corrected_weather).et0[3])


def test_adapter_reads_corrected_fields_from_store_path(tmp_path: Path):
    weather_store(tmp_path, ('TMP_corr', 'PRATE_corr'))
    actual = get_weather_data(
        store=get_store(tmp_path / 'weather'),
        latitude=50.8,
        longitude=4.3,
        altitude=100,
        start=date(2020, 2, 28),
        end=date(2020, 3, 2),
    )
    np.testing.assert_allclose(actual.pr, [12, np.nan, 0, 1])
    assert np.isnan(actual.et0[1])
    assert actual.et0[0] == pytest.approx(
        _et0_fao56_for_uccle(t_mean=23, rs0=0.752 * extraterrestrial_radiation(50.8, 59))
    )


def test_adapter_ignores_only_pristine_empty_correction_group(tmp_path: Path):
    root = weather_store(tmp_path)
    correction = root.require_group('correction')
    actual = get_weather_data(
        store=get_store(tmp_path / 'weather'),
        latitude=50.8,
        longitude=4.3,
        altitude=100,
        start=date(2020, 2, 28),
        end=date(2020, 3, 2),
    )
    np.testing.assert_allclose(actual.pr, [8, np.nan, 0, 2])
    correction.attrs['write_state'] = 'writing'
    with pytest.raises(StorageError, match='completed'):
        get_weather_data(
            store=get_store(tmp_path / 'weather'),
            latitude=50.8,
            longitude=4.3,
            altitude=100,
            start=date(2020, 2, 28),
            end=date(2020, 3, 2),
        )


def test_adapter_rejects_array_in_place_of_correction_group(tmp_path: Path):
    root = weather_store(tmp_path)
    root.create_array('correction', shape=(1,), dtype='i4')
    with pytest.raises(TypeError, match='must be a Zarr group'):
        read_weather(root)
