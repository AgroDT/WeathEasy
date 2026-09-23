from collections.abc import Callable
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np


class VarInfo(NamedTuple):
    en: str
    ru: str


class Cfs2Variable(NamedTuple):
    daily_stat: Callable
    info: VarInfo


ONE_DAY = timedelta(days=1)

CFS2_DIR = 'cfs2'
CFS2_KEY_UPDATED = 'updated'
CFS2_HHS = '00', '06', '12', '18'

CFS2_REANALYSIS_DIR = CFS2_DIR + '/reanalysis'
CFS2_REANALYSIS_FIRST_DATE = date(2011, 4, 1)
CFS2_REANALYSIS_LAST_DATE_OFFSET = timedelta(days=3)

CFS2_FORECAST_DIR = CFS2_DIR + '/forecast'
CFS2_FORECAST_DAYS = timedelta(days=180)


CFS2_FLX_PARAMS = {
    'var_DLWRF': 'on',
    'var_DSWRF': 'on',
    'var_GFLUX': 'on',
    'var_LHTFL': 'on',
    'var_PRATE': 'on',
    'var_PEVPR': 'on',
    'var_PRES': 'on',
    'var_QMAX': 'on',
    'var_QMIN': 'on',
    'var_SHTFL': 'on',
    'var_SNOD': 'on',
    'var_SOILW': 'on',
    'var_SPFH': 'on',
    'var_TMAX': 'on',
    'var_TMIN': 'on',
    'var_TMP': 'on',
    'var_UGRD': 'on',
    'var_ULWRF': 'on',
    'var_USWRF': 'on',
    'var_VGRD': 'on',
    'var_WEASD': 'on',
    'lev_0-0.1_m_below_ground': 'on',
    'lev_0.1-0.4_m_below_ground': 'on',
    'lev_0.4-1_m_below_ground': 'on',
    'lev_1-2_m_below_ground': 'on',
    'lev_2_m_above_ground': 'on',
    'lev_10_m_above_ground': 'on',
    'lev_surface': 'on',
}

# fmt: off
CFS2_FLX_BANDS = {
    'PEVPR': Cfs2Variable(np.mean, VarInfo(
        en='Potential evaporation, mm/day (four-sample approximation)',
        ru='Приближённое суточное потенциальное испарение, мм/сутки',
    )),
    'DLWRF': Cfs2Variable(np.mean, VarInfo(
        en='Downward Long-Wave Radiaiton Flux, W/m²',
        ru='Нисходящий поток инфракрасного излучения, Вт/м²',
    )),
    'DSWRF': Cfs2Variable(np.mean, VarInfo(
        en='Downward Short-Wave Radiation Flux, W/m²',
        ru='Нисходящий поток коротковолнового солнечного излучения, Вт/м²',
    )),
    'GFLUX': Cfs2Variable(np.mean, VarInfo(
        en='Ground Heat Flux, W/m²',
        ru='Поток тепла через поверхность почвы, Вт/м²',
    )),
    'LHTFL': Cfs2Variable(np.mean, VarInfo(
        en='Latent Heat Net Flux, W/m²',
        ru='Поток скрытого тепла, Вт/м²',
    )),
    'PRATE': Cfs2Variable(np.sum, VarInfo(
        en='Precipitation Rate, kg/m²/s',
        ru='Интенсивность осадков, кг/м²/с',
    )),
    'PRES': Cfs2Variable(np.mean, VarInfo(
        en='Atmospheric Pressure, Pa',
        ru='Атмосферное давление, Па',
    )),
    'QMAX': Cfs2Variable(np.max, VarInfo(
        en='Maximum Specific Humidity, kg/kg',
        ru='Максимальная удельная влажность, кг/кг',
    )),
    'QMIN': Cfs2Variable(np.min, VarInfo(
        en='Minimum Specific Humidity, kg/kg',
        ru='Минимальная удельная влажность, кг/кг',
    )),
    'SHTFL': Cfs2Variable(np.mean, VarInfo(
        en='Sensible Heat Net Flux, W/m²',
        ru='Сетевой поток явного тепла, Вт/м²',
    )),
    'SNOD': Cfs2Variable(np.max, VarInfo(
        en='Snow Depth, m',
        ru='Глубина снежного покрова, м',
    )),
    'SOILW_0-0.1m': Cfs2Variable(np.max, VarInfo(
        en='Volumetric Soil Moisture Content, m³/m³',
        ru='Объемное содержание влаги в почве, м³/м³',
    )),
    'SOILW_0.1-0.4m': Cfs2Variable(np.max, VarInfo(
        en='',
        ru='',
    )),
    'SOILW_0.4-1m': Cfs2Variable(np.max, VarInfo(
        en='',
        ru='',
    )),
    'SOILW_1-2m': Cfs2Variable(np.max, VarInfo(
        en='',
        ru='',
    )),
    'SPFH': Cfs2Variable(np.mean, VarInfo(
        en='Specific Humidity, kg/kg',
        ru='Удельная влажность, кг/кг',
    )),
    'TMAX': Cfs2Variable(np.max, VarInfo(
        en='Maximum Temperature, °C',
        ru='Максимальная температура, °C',
    )),
    'TMIN': Cfs2Variable(np.min, VarInfo(
        en='Minimum Temperature, °C',
        ru='Минимальная температура, °C',
    )),
    'TMP': Cfs2Variable(np.mean, VarInfo(
        en='Temperature, °C',
        ru='Температура, °C',
    )),
    'TMP_0m': Cfs2Variable(np.mean, VarInfo(
        en='Soil temperature, °C',
        ru='Температура почвы, °C',
    )),
    'TMP_0_0.1m': Cfs2Variable(np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_0.1_0.4m': Cfs2Variable(np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_0.4_1m': Cfs2Variable(np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_1-2m': Cfs2Variable(np.mean, VarInfo(
        en='',
        ru='',
    )),
    'UGRD': Cfs2Variable(np.mean, VarInfo(
        en='U-component of Wind, m/s',
        ru='Зональная составляющая ветра, т.е. компонент ветра по оси запад-восток, м/с',
    )),
    'ULWRF': Cfs2Variable(np.mean, VarInfo(
        en='Upward Long-Wave Radiation Flux, W/m²',
        ru='Восходящий поток инфракрасного излучения, Вт/м²',
    )),
    'USWRF': Cfs2Variable(np.mean, VarInfo(
        en='Upward Short-Wave Radiation Flux, W/m²',
        ru='Восходящий поток коротковолнового солнечного излучения, Вт/м²',
    )),
    'VGRD': Cfs2Variable(np.mean, VarInfo(
        en='V-component of Wind, m/s',
        ru='Меридиональная составляющая ветра, т.е. компонент ветра по оси юг-север, м/с',
    )),
    'WEASD': Cfs2Variable(np.sum, VarInfo(
        en='Water Equivalent of Accumulated Snow Depth, kg/m²',
        ru='Водный эквивалент накопленной глубины снега, кг/м²',
    )),
}
# fmt: on


CFS2_PGB_PARAMS = {
    'var_RH': 'on',
    'lev_2_m_above_ground': 'on',
}

# fmt: off
CFS2_PGB_BANDS = {
    'RH': Cfs2Variable(np.mean, VarInfo(
        en='Relative Humidity , %',
        ru='Относительная влажность, %',
    )),
}
# fmt: on

CFS2_BANDS = {**CFS2_FLX_BANDS, **CFS2_PGB_BANDS}

CMIP6_DIR = 'cmip6'
CMIP6_FIRST_YEAR = 1950
CMIP6_LAST_YEAR = 2100
CMIP6_LAST_HISTORICAL_YEAR = 2014

CMIP6_VARS = {
    'hurs': VarInfo(
        en='Near-surface relative humidity, %',
        ru='',
    ),
    'huss': VarInfo(
        en='Mass fraction	Near-surface specific humidity',
        ru='',
    ),
    'pr': VarInfo(
        en='Precipitation (mean of the daily precipitation rate), kg/m²/s',
        ru='',
    ),
    'rlds': VarInfo(
        en='Surface downwelling longwave radiation, W/m²',
        ru='',
    ),
    'rsds': VarInfo(
        en='Surface downwelling shortwave radiation, W/m²',
        ru='',
    ),
    'sfcWind': VarInfo(
        en='Daily-mean near-surface wind speed, m/s',
        ru='',
    ),
    'tas': VarInfo(
        en='Daily near-surface air temperature, K',
        ru='',
    ),
    'tasmin': VarInfo(
        en='Daily minimum near-surface air temperature, K',
        ru='',
    ),
    'tasmax': VarInfo(
        en='Daily maximum near-surface air temperature, K',
        ru='',
    ),
}


ERA5_DIR = 'era5'
ERA5_RESOLUTION = 0.1
ERA5_FIRST_YEAR = 1950
# Edge-based BoundingBox: centers run from (-180, -90) to (179.9, 90) in steps of 0.1°


class Era5Var(NamedTuple):
    cds_name: str  # CDS API variable name
    nc_name: str  # NetCDF short name in downloaded file
    accumulated: bool  # True: daily value = 00:00 UTC of the following day × scale
    daily_stat: Callable | None  # for instantaneous: np.mean / np.max / np.min
    scale: float  # unit conversion multiplier
    offset: float  # unit conversion offset (applied after scale)
    info: VarInfo


ERA5_VARS: dict[str, Era5Var] = {
    'TMP': Era5Var(
        cds_name='2m_temperature',
        nc_name='t2m',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Temperature, °C', ru='Температура, °C'),
    ),
    'TMAX': Era5Var(
        cds_name='2m_temperature',
        nc_name='t2m',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Maximum Temperature, °C', ru='Максимальная температура, °C'),
    ),
    'TMIN': Era5Var(
        cds_name='2m_temperature',
        nc_name='t2m',
        accumulated=False,
        daily_stat=np.min,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Minimum Temperature, °C', ru='Минимальная температура, °C'),
    ),
    'UGRD': Era5Var(
        cds_name='10m_u_component_of_wind',
        nc_name='u10',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=0.0,
        info=VarInfo(en='U-component of Wind, m/s', ru='Зональная составляющая ветра, м/с'),
    ),
    'VGRD': Era5Var(
        cds_name='10m_v_component_of_wind',
        nc_name='v10',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=0.0,
        info=VarInfo(en='V-component of Wind, m/s', ru='Меридиональная составляющая ветра, м/с'),
    ),
    'PRES': Era5Var(
        cds_name='surface_pressure',
        nc_name='sp',
        accumulated=False,
        daily_stat=np.mean,
        scale=0.01,
        offset=0.0,
        info=VarInfo(en='Atmospheric Pressure, hPa', ru='Атмосферное давление, гПа'),
    ),
    'PRATE': Era5Var(
        cds_name='total_precipitation',
        nc_name='tp',
        accumulated=True,
        daily_stat=None,
        scale=1000.0 / 86400,
        offset=0.0,
        info=VarInfo(en='Precipitation Rate, kg/m²/s', ru='Интенсивность осадков, кг/м²/с'),
    ),
    'SNOD': Era5Var(
        cds_name='snow_depth',
        nc_name='sde',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=0.0,
        info=VarInfo(en='Snow Depth, m', ru='Глубина снежного покрова, м'),
    ),
    'WEASD': Era5Var(
        cds_name='snow_depth_water_equivalent',
        nc_name='sd',
        accumulated=False,
        daily_stat=np.max,
        scale=1000.0,
        offset=0.0,
        info=VarInfo(
            en='Water Equivalent of Accumulated Snow Depth, kg/m²',
            ru='Водный эквивалент накопленной глубины снега, кг/м²',
        ),
    ),
    'SOILW_0-0.1m': Era5Var(
        cds_name='volumetric_soil_water_layer_1',
        nc_name='swvl1',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=0.0,
        info=VarInfo(
            en='Volumetric Soil Moisture 0–0.07 m, m³/m³', ru='Влажность почвы 0–0.07 м, м³/м³'
        ),
    ),
    'SOILW_0.1-0.4m': Era5Var(
        cds_name='volumetric_soil_water_layer_2',
        nc_name='swvl2',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=0.0,
        info=VarInfo(
            en='Volumetric Soil Moisture 0.07–0.28 m, m³/m³',
            ru='Влажность почвы 0.07–0.28 м, м³/м³',
        ),
    ),
    'SOILW_0.4-1m': Era5Var(
        cds_name='volumetric_soil_water_layer_3',
        nc_name='swvl3',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=0.0,
        info=VarInfo(
            en='Volumetric Soil Moisture 0.28–1 m, m³/m³', ru='Влажность почвы 0.28–1 м, м³/м³'
        ),
    ),
    'SOILW_1-2m': Era5Var(
        cds_name='volumetric_soil_water_layer_4',
        nc_name='swvl4',
        accumulated=False,
        daily_stat=np.max,
        scale=1.0,
        offset=0.0,
        info=VarInfo(
            en='Volumetric Soil Moisture 1–2.89 m, m³/m³', ru='Влажность почвы 1–2.89 м, м³/м³'
        ),
    ),
    'TMP_0m': Era5Var(
        cds_name='skin_temperature',
        nc_name='skt',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Skin Temperature, °C', ru='Температура поверхности, °C'),
    ),
    'TMP_0_0.1m': Era5Var(
        cds_name='soil_temperature_level_1',
        nc_name='stl1',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Soil Temperature 0–0.07 m, °C', ru='Температура почвы 0–0.07 м, °C'),
    ),
    'TMP_0.1_0.4m': Era5Var(
        cds_name='soil_temperature_level_2',
        nc_name='stl2',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Soil Temperature 0.07–0.28 m, °C', ru='Температура почвы 0.07–0.28 м, °C'),
    ),
    'TMP_0.4_1m': Era5Var(
        cds_name='soil_temperature_level_3',
        nc_name='stl3',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Soil Temperature 0.28–1 m, °C', ru='Температура почвы 0.28–1 м, °C'),
    ),
    'TMP_1-2m': Era5Var(
        cds_name='soil_temperature_level_4',
        nc_name='stl4',
        accumulated=False,
        daily_stat=np.mean,
        scale=1.0,
        offset=-273.15,
        info=VarInfo(en='Soil Temperature 1–2.89 m, °C', ru='Температура почвы 1–2.89 м, °C'),
    ),
    'LHTFL': Era5Var(
        cds_name='surface_latent_heat_flux',
        nc_name='slhf',
        accumulated=True,
        daily_stat=None,
        scale=-1.0 / 86400,
        offset=0.0,
        info=VarInfo(en='Latent Heat Net Flux, W/m²', ru='Поток скрытого тепла, Вт/м²'),
    ),
    'SHTFL': Era5Var(
        cds_name='surface_sensible_heat_flux',
        nc_name='sshf',
        accumulated=True,
        daily_stat=None,
        scale=-1.0 / 86400,
        offset=0.0,
        info=VarInfo(en='Sensible Heat Net Flux, W/m²', ru='Поток явного тепла, Вт/м²'),
    ),
    'DSWRF': Era5Var(
        cds_name='surface_solar_radiation_downwards',
        nc_name='ssrd',
        accumulated=True,
        daily_stat=None,
        scale=1.0 / 86400,
        offset=0.0,
        info=VarInfo(
            en='Downward Short-Wave Radiation Flux, W/m²',
            ru='Нисходящий поток коротковолнового солнечного излучения, Вт/м²',
        ),
    ),
    'DLWRF': Era5Var(
        cds_name='surface_thermal_radiation_downwards',
        nc_name='strd',
        accumulated=True,
        daily_stat=None,
        scale=1.0 / 86400,
        offset=0.0,
        info=VarInfo(
            en='Downward Long-Wave Radiation Flux, W/m²',
            ru='Нисходящий поток инфракрасного излучения, Вт/м²',
        ),
    ),
    'PEVPR': Era5Var(
        cds_name='potential_evaporation',
        nc_name='pev',
        accumulated=True,
        daily_stat=None,
        scale=-1000.0,
        offset=0.0,
        info=VarInfo(
            en='Potential open-water evaporation, mm/day (positive upwards)',
            ru='Потенциальное испарение открытой водной поверхности, мм/сутки',
        ),
    ),
}

# Variables derived from multiple ERA5 inputs (computed in download.py)
ERA5_DERIVED_INFO: dict[str, VarInfo] = {
    'WIND': VarInfo(
        en='Daily mean wind speed at 10 m, m/s', ru='Средняя скорость ветра на 10 м, м/с'
    ),
    'RH': VarInfo(en='Relative Humidity, %', ru='Относительная влажность, %'),
    'SPFH': VarInfo(en='Specific Humidity, kg/kg', ru='Удельная влажность, кг/кг'),
    'QMAX': VarInfo(
        en='Maximum Specific Humidity, kg/kg', ru='Максимальная удельная влажность, кг/кг'
    ),
    'QMIN': VarInfo(
        en='Minimum Specific Humidity, kg/kg', ru='Минимальная удельная влажность, кг/кг'
    ),
    'USWRF': VarInfo(
        en='Upward Short-Wave Radiation Flux, W/m²',
        ru='Восходящий поток коротковолнового солнечного излучения, Вт/м²',
    ),
    'ULWRF': VarInfo(
        en='Upward Long-Wave Radiation Flux, W/m²',
        ru='Восходящий поток инфракрасного излучения, Вт/м²',
    ),
}

ERA5_ALL_VARS: dict[str, VarInfo] = {
    **{k: v.info for k, v in ERA5_VARS.items()},
    **ERA5_DERIVED_INFO,
}
