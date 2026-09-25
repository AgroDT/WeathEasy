"""Agroclimate summaries of an already selected daily weather interval.

Input units: temperature °C, humidity %, precipitation and potential evaporation
mm/day, latent heat flux W/m² (positive upward), soil moisture m³/m³. Date
arguments describe the selected interval; arrays have no time index and are not
sliced here. NaN propagates through numerical summaries; an unknown day breaks a
dry spell, so the reported spell is the longest confirmed run of dry days.
"""

# Date arguments remain part of the donor API; callers select the interval.
# ruff: noqa: ARG002

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date

    from numpy.typing import NDArray


class AgroClimateCalculator:
    def __init__(self, data: Mapping[str, NDArray]) -> None:
        self.data = data

    def _values(self, variable: str) -> NDArray[np.float64]:
        if variable not in self.data:
            msg = f'Missing weather variable: {variable}'
            raise ValueError(msg)
        values = np.asarray(self.data[variable], dtype=np.float64)
        if values.ndim != 1 or not values.size:
            msg = f'{variable} must contain a nonempty daily series'
            raise ValueError(msg)
        return values

    def actual_evapotranspiration_LE(  # noqa: N802
        self, start_date: date, end_date: date
    ) -> float:
        """Return mean actual evapotranspiration in mm/day from latent heat flux."""
        return float(np.mean(self._values('LHTFL') * 86400 / 2.45e6))

    def soil_water_deficit(self, start_date: date, end_date: date) -> float:
        """Return mean potential minus actual evaporation in mm/day."""
        potential = self._values('PEVPR')
        actual = self._values('LHTFL') * 86400 / 2.45e6
        if potential.shape != actual.shape:
            msg = 'PEVPR and LHTFL must cover the same daily interval'
            raise ValueError(msg)
        return float(np.mean(potential - actual))

    def mean_soil_water_content(
        self, start_date: date, end_date: date, depth: str = 'SOILW_0-0.1m'
    ) -> float:
        """Return mean volumetric soil moisture as a percentage."""
        return float(np.mean(self._values(depth)) * 100)

    def average_temperature(self, start_date: date, end_date: date) -> float:
        return float(np.mean(self._values('TMP')))

    def average_humidity(self, start_date: date, end_date: date) -> float:
        return float(np.mean(self._values('RH')))

    def total_precipitation(self, start_date: date, end_date: date) -> float:
        """Return total precipitation in mm; input is daily precipitation in mm/day."""
        return float(np.sum(self._values('PRATE')))

    def max_no_rain_period(self, start_date: date, end_date: date, threshold: float = 1.0) -> int:
        """Return the longest confirmed dry spell in days (threshold in mm/day)."""
        longest = current = 0
        for is_dry in self._values('PRATE') < threshold:
            current = current + 1 if is_dry else 0
            longest = max(longest, current)
        return longest

    def max_daily_precipitation(self, start_date: date, end_date: date) -> float:
        return float(np.max(self._values('PRATE')))

    def calculate_all_parameters(self, start_date: date, end_date: date) -> dict[str, float | int]:
        return {
            'AET': self.actual_evapotranspiration_LE(start_date, end_date),
            'SWD': self.soil_water_deficit(start_date, end_date),
            'SWC': self.mean_soil_water_content(start_date, end_date),
            'T_AVG': self.average_temperature(start_date, end_date),
            'RH_AVG': self.average_humidity(start_date, end_date),
            'P_SUM': self.total_precipitation(start_date, end_date),
            'P0_INT': self.max_no_rain_period(start_date, end_date),
            'P_MAX': self.max_daily_precipitation(start_date, end_date),
        }


def format_results(results: Mapping[str, float | int]) -> str:
    return '\n'.join(
        [
            f"Фактическая эвапотранспирация (AET): {results['AET']:.2f} мм/день",
            f"Дефицит влаги в почве (SWD): {results['SWD']:.2f} мм/день",
            f"Среднее содержание влаги в почве (SWC): {results['SWC']:.2f} %",
            f"Средняя температура воздуха (T_AVG): {results['T_AVG']:.2f} °C",
            f"Средняя относительная влажность воздуха (RH_AVG): {results['RH_AVG']:.2f} %",
            f"Сумма осадков (P_SUM): {results['P_SUM']:.2f} мм",
            "Максимальная продолжительность периода без осадков (P0_INT): "
            f"{results['P0_INT']} дней",
            f"Максимальное количество осадков за сутки (P_MAX): {results['P_MAX']:.2f} мм",
        ]
    )
