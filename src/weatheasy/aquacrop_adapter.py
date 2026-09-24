"""Read prepared ERA5 weather for AquaCrop and calculate daily FAO-56 ET0.

No model loading or downloading is performed. Tmin/Tmax remain uncorrected;
the selected TMP/TMP_corr is the mean temperature used in ET0. This preserves
the donor's choice instead of FAO-56's default (Tmin + Tmax) / 2.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import zarr

from weatheasy import Coords, get_era5_data


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


WeatherRecord = np.dtype(
    [
        ('tmin', np.float64),
        ('tmax', np.float64),
        ('pr', np.float64),
        ('et0', np.float64),
    ]
)


def saturation_vapour_pressure(temperature: ArrayLike) -> NDArray[np.float64]:
    """FAO-56 equation 11: saturation vapour pressure (kPa), temperature in °C."""
    t = np.asarray(temperature, dtype=np.float64)
    return np.asarray(0.6108 * np.exp(17.27 * t / (t + 237.3)))


def extraterrestrial_radiation(latitude: float, day_of_year: ArrayLike) -> NDArray[np.float64]:
    """Daily Ra (MJ/m²/day), equations 21–25; latitude in degrees.

    Clipping the sunset cosine handles polar night/day. FAO-56 notes limited
    accuracy of these equations above 55° latitude in winter.
    """
    angle = 2 * np.pi * np.asarray(day_of_year, dtype=np.float64) / 365
    distance = 1 + 0.033 * np.cos(angle)
    declination = 0.409 * np.sin(angle - 1.39)
    lat = np.radians(latitude)
    sunset = np.arccos(np.clip(-np.tan(lat) * np.tan(declination), -1, 1))
    radiation = (
        (24 * 60 / np.pi)
        * 0.0820
        * distance
        * (
            sunset * np.sin(lat) * np.sin(declination)
            + np.cos(lat) * np.cos(declination) * np.sin(sunset)
        )
    )
    return np.asarray(np.maximum(radiation, 0))


def net_longwave_radiation(
    t_min: ArrayLike, t_max: ArrayLike, ea: ArrayLike, rs: ArrayLike, rs0: ArrayLike
) -> NDArray[np.float64]:
    """Equation 39: Rnl (MJ/m²/day); °C, kPa and MJ/m²/day inputs.

    The agreed cloud factor is limited to [0.05, 1]. A nonpositive Rs0 uses
    1e-6 in the denominator; missing Rs0 remains missing.
    """
    low, high, vapour, solar, clear = (
        np.asarray(value, dtype=np.float64) for value in (t_min, t_max, ea, rs, rs0)
    )
    denominator = np.where(clear <= 0, 1e-6, clear)
    cloud = np.clip(1.35 * solar / denominator - 0.35, 0.05, 1)
    return np.asarray(
        4.903e-9
        * ((high + 273.16) ** 4 + (low + 273.16) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(vapour))
        * cloud
    )


def et0_fao56(
    *,
    t_min: ArrayLike,
    t_max: ArrayLike,
    t_mean: ArrayLike,
    rh_mean: ArrayLike,
    wind: ArrayLike,
    pressure: ArrayLike,
    rs: ArrayLike,
    rs0: ArrayLike,
    wind_height: float = 10,
) -> NDArray[np.float64]:
    """Daily equation 6 with G=0, returning ET0 in mm/day.

    Inputs: temperatures in °C, RH in %, wind in m/s at wind_height metres,
    pressure in kPa, Rs/Rs0 in MJ/m²/day. Arrays broadcast; scalars produce a
    zero-dimensional array. t_mean is explicit so a corrected mean can be used
    without changing Tmin/Tmax. NaN propagates; nonpositive ET0 is returned as
    calculated, with the AquaCrop adapter applying its output policy separately.
    """
    low, high, mean, humidity, speed, press, solar = (
        np.asarray(value, dtype=np.float64)
        for value in (t_min, t_max, t_mean, rh_mean, wind, pressure, rs)
    )
    es = (saturation_vapour_pressure(low) + saturation_vapour_pressure(high)) / 2
    ea = humidity / 100 * es
    delta = 4098 * saturation_vapour_pressure(mean) / (mean + 237.3) ** 2
    gamma = 0.000665 * press
    wind2 = speed * 4.87 / np.log(67.8 * wind_height - 5.42)
    net_radiation = 0.77 * solar - net_longwave_radiation(low, high, ea, solar, rs0)
    numerator = 0.408 * delta * net_radiation + gamma * 900 / (mean + 273) * wind2 * (es - ea)
    return np.asarray(numerator / (delta + gamma * (1 + 0.34 * wind2)))


def get_weather_data(
    *,
    root: zarr.Group,
    latitude: float,
    longitude: float,
    altitude: float,
    start: date,
    end: date,
    use_corrected: bool = True,
) -> np.recarray:
    """Return inclusive daily tmin/tmax (°C), pr/et0 (mm/day) records.

    altitude is elevation above sea level in metres, used only for clear-sky
    radiation. WIND is the prepared mean hourly speed at 10 m. Corrected mean
    temperature/precipitation are selected independently when their arrays exist;
    missing days in an existing corrected array remain NaN. Raw precipitation is
    converted from kg/m²/s; PRATE_corr already contains mm/day. The donor's 1 mm
    precipitation threshold and nonpositive ET0 -> NaN policy are preserved.
    """
    available: set[str] = set()
    if use_corrected and 'era5/correction' in root:
        correction = root['era5/correction']
        if not isinstance(correction, zarr.Group):
            msg = 'era5/correction must be a Zarr group'
            raise TypeError(msg)
        available = set(correction.array_keys())
    tmp_var = 'TMP_corr' if 'TMP_corr' in available else 'TMP'
    prate_var = 'PRATE_corr' if 'PRATE_corr' in available else 'PRATE'
    raw = get_era5_data(
        root=root,
        begin=start,
        end=end,
        coords=Coords(latitude, longitude),
        variables=['TMIN', 'TMAX', 'WIND', 'PRES', 'RH', 'DSWRF', tmp_var, prate_var],
    ).astype(np.float64)
    t_min, t_max, wind, pressure, humidity, solar, mean, precip = raw
    if prate_var == 'PRATE':
        precip = precip * 86400
    days = np.array(
        [(start + timedelta(days=index)).timetuple().tm_yday for index in range(raw.shape[1])]
    )
    rs0 = (0.75 + 2e-5 * altitude) * extraterrestrial_radiation(latitude, days)
    et0 = et0_fao56(
        t_min=t_min,
        t_max=t_max,
        t_mean=mean,
        rh_mean=humidity,
        wind=wind,
        pressure=pressure * 0.1,
        rs=solar * 0.0864,
        rs0=rs0,
    )
    result = np.empty(raw.shape[1], dtype=WeatherRecord).view(np.recarray)
    result.tmin = t_min
    result.tmax = t_max
    result.pr = np.where(np.isnan(precip), np.nan, np.where(precip >= 1, precip, 0))
    result.et0 = np.where(et0 > 0, et0, np.nan)
    return result
