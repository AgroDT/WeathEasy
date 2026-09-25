from datetime import date

import numpy as np
import pytest

from weatheasy.agro_clim import AgroClimateCalculator, format_results


BEGIN, END = date(2020, 2, 28), date(2020, 3, 2)


def test_summaries_units_and_formatter():
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
    output = format_results(result)
    assert '(AET): 3.53 мм/день' in output
    assert '(SWD): 1.47 мм/день' in output
    assert '(SWC): 20.00 %' in output
    assert '(P0_INT): 2 дней' in output
    # The caller provides the selected interval; date arguments do not silently slice arrays.
    assert calculator.total_precipitation(BEGIN, BEGIN) == 9.5


def test_missing_days_and_negative_evaporation():
    calculator = AgroClimateCalculator(
        {
            'LHTFL': np.array([-100, 0]),
            'PEVPR': np.array([0, 2]),
            'PRATE': np.array([0, 0, np.nan, 0, 0, 0, 1]),
        }
    )
    assert calculator.actual_evapotranspiration_LE(BEGIN, END) == pytest.approx(-1.7632653061)
    assert calculator.soil_water_deficit(BEGIN, END) == pytest.approx(2.7632653061)
    assert np.isnan(calculator.total_precipitation(BEGIN, END))
    assert np.isnan(calculator.max_daily_precipitation(BEGIN, END))
    assert calculator.max_no_rain_period(BEGIN, END) == 3
    assert calculator.max_no_rain_period(BEGIN, END, threshold=0) == 0
    calculator = AgroClimateCalculator({'LHTFL': np.array([100, np.nan])})
    assert np.isnan(calculator.actual_evapotranspiration_LE(BEGIN, END))


def test_invalid_calculator_inputs():
    with pytest.raises(ValueError, match='Missing weather variable: TMP'):
        AgroClimateCalculator({}).average_temperature(BEGIN, END)
    with pytest.raises(ValueError, match='nonempty daily series'):
        AgroClimateCalculator({'TMP': np.array([])}).average_temperature(BEGIN, END)
    with pytest.raises(ValueError, match='same daily interval'):
        AgroClimateCalculator(
            {'LHTFL': np.array([100, 200]), 'PEVPR': np.array([4])}
        ).soil_water_deficit(BEGIN, END)
