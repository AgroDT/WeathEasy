from datetime import date

import numpy as np
import pytest

from weatheasy.agro_clim import AgroClimateCalculator


BEGIN, END = date(2020, 2, 28), date(2020, 3, 2)


@pytest.fixture(name='evaporation_calculator')
def create_evaporation_calculator():
    return AgroClimateCalculator({'LHTFL': np.array([-100, 0]), 'PEVPR': np.array([0, 2])})


@pytest.fixture(name='precipitation_calculator')
def create_precipitation_calculator():
    return AgroClimateCalculator({'PRATE': np.array([0, 0, np.nan, 0, 0, 0, 1])})


def test_summaries_and_units():
    calculator = AgroClimateCalculator(
        {
            'LHTFL': np.array([50, 100, 150, 100]),
            'PEVPR': np.array([4, 6, 6, 4]),
            'SOILW_0-0.1m': np.array([0.1, 0.2, 0.3, 0.2]),
            'TMP': np.array([0, 5, 15, 20]),
            'RH': np.array([50, 60, 70, 60]),
            'PRATE': np.array([0, 0.5, 1, 8]),
        }
    )
    result = calculator.calculate_all_parameters(BEGIN, END)
    assert result == pytest.approx(
        {
            'AET': 3.526530612244898,
            'SWD': 1.473469387755102,
            'SWC': 20,
            'T_AVG': 10,
            'RH_AVG': 60,
            'P_SUM': 9.5,
            'P0_INT': 2,
            'P_MAX': 8,
        }
    )


def test_date_arguments_do_not_slice_selected_precipitation():
    calculator = AgroClimateCalculator({'PRATE': np.array([0, 0.5, 1, 8])})
    assert calculator.total_precipitation(BEGIN, BEGIN) == 9.5


def test_negative_actual_evaporation(evaporation_calculator: AgroClimateCalculator):
    calculator = evaporation_calculator
    assert calculator.actual_evapotranspiration_le(BEGIN, END) == pytest.approx(-1.7632653061)


def test_water_deficit_includes_negative_evaporation(evaporation_calculator: AgroClimateCalculator):
    calculator = evaporation_calculator
    assert calculator.soil_water_deficit(BEGIN, END) == pytest.approx(2.7632653061)


@pytest.mark.parametrize('method', ['total_precipitation', 'max_daily_precipitation'])
def test_missing_precipitation_propagates(
    precipitation_calculator: AgroClimateCalculator, method: str
):
    assert np.isnan(getattr(precipitation_calculator, method)(BEGIN, END))


def test_missing_day_breaks_dry_period(precipitation_calculator: AgroClimateCalculator):
    calculator = precipitation_calculator
    assert calculator.max_no_rain_period(BEGIN, END) == 3


def test_zero_threshold_excludes_dry_days(precipitation_calculator: AgroClimateCalculator):
    calculator = precipitation_calculator
    assert calculator.max_no_rain_period(BEGIN, END, threshold=0) == 0


def test_missing_latent_heat_propagates_to_evaporation():
    calculator = AgroClimateCalculator({'LHTFL': np.array([100, np.nan])})
    assert np.isnan(calculator.actual_evapotranspiration_le(BEGIN, END))


def test_temperature_rejects_missing_variable():
    with pytest.raises(ValueError, match='Missing weather variable: TMP'):
        AgroClimateCalculator({}).average_temperature(BEGIN, END)


def test_temperature_rejects_empty_series():
    with pytest.raises(ValueError, match='nonempty daily series'):
        AgroClimateCalculator({'TMP': np.array([])}).average_temperature(BEGIN, END)


def test_water_deficit_rejects_different_intervals():
    with pytest.raises(ValueError, match='same daily interval'):
        AgroClimateCalculator(
            {'LHTFL': np.array([100, 200]), 'PEVPR': np.array([4])}
        ).soil_water_deficit(BEGIN, END)
