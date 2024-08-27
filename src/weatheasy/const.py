from collections.abc import Callable
from datetime import date, timedelta
from typing import NamedTuple

import numpy as np
from rasterio.coords import BoundingBox


class VarInfo(NamedTuple):
    en: str
    ru: str


class Cfs2Band(NamedTuple):
    forecast: int
    reanalysis: int
    daily_stat: Callable
    info: VarInfo


ONE_DAY = timedelta(days=1)

CFS2_DIR = 'cfs2'
CFS2_KEY_UPDATED = 'updated'
CFS2_HHS = '00', '06', '12', '18'

CFS2_REANALYSIS_DIR = CFS2_DIR + '/reanalysis'
CFS2_REANALYSIS_FIRST_DATE = date(2011, 4, 1)
CFS2_REANALYSIS_LAST_DATE_OFFSET = timedelta(days=3)
CFS2_REANALYSIS_RESOLUTION = 0.5, 0.5
CFS2_REANALYSIS_BBOX = BoundingBox(-180.25, -90.25, 179.75, 90.25)

CFS2_FORECAST_DIR = CFS2_DIR + '/forecast'
CFS2_FORECAST_DAYS = timedelta(days=180)

CFS2_FLX_RESOLUTION = 0.9374986945169713, 0.9473684210526315
CFS2_FLX_BBOX = BoundingBox(
    -0.46874934725848566, -90.24931578947368, 359.5307493472585, 89.75068421052632
)

CFS2_FLX_PARAMS = {
    'var_DLWRF': 'on',
    'var_DSWRF': 'on',
    'var_GFLUX': 'on',
    'var_LHTFL': 'on',
    'var_PRATE': 'on',
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
    'DLWRF': Cfs2Band(9, 571, np.mean, VarInfo(
        en='Downward Long-Wave Radiaiton Flux, W/m²',
        ru='Нисходящий поток инфракрасного излучения, Вт/м²',
    )),
    'DSWRF': Cfs2Band(12, 576, np.mean, VarInfo(
        en='Downward Short-Wave Radiation Flux, W/m²',
        ru='Нисходящий поток коротковолнового солнечного излучения, Вт/м²',
    )),
    'GFLUX': Cfs2Band(14, 644, np.mean, VarInfo(
        en='Ground Heat Flux, W/m²',
        ru='Поток тепла через поверхность почвы, Вт/м²',
    )),
    'LHTFL': Cfs2Band(2, 564, np.mean, VarInfo(
        en='Latent Heat Net Flux, W/m²',
        ru='Поток скрытого тепла, Вт/м²',
    )),
    'PRATE': Cfs2Band(13, 591, np.sum, VarInfo(
        en='Precipitation Rate, kg/m²/s',
        ru='Интенсивность осадков, кг/м²/с',
    )),
    'PRES': Cfs2Band(19, 600, np.mean, VarInfo(
        en='Atmospheric Pressure, hPa',
        ru='Атмосферное давление, гПа',
    )),
    'QMAX': Cfs2Band(22, 603, np.max, VarInfo(
        en='Maximum Specific Humidity, kg/kg',
        ru='Максимальная удельная влажность, кг/кг',
    )),
    'QMIN': Cfs2Band(23, 604, np.min, VarInfo(
        en='Minimum Specific Humidity, kg/kg',
        ru='Минимальная удельная влажность, кг/кг',
    )),
    'SHTFL': Cfs2Band(1, 563, np.mean, VarInfo(
        en='Sensible Heat Net Flux, W/m²',
        ru='Сетевой поток явного тепла, Вт/м²',
    )),
    'SNOD': Cfs2Band(28, 625, np.max, VarInfo(
        en='Snow Depth, m',
        ru='Глубина снежного покрова, м',
    )),
    'SOILW_0-0.1m': Cfs2Band(4, 566, np.max, VarInfo(
        en='Volumetric Soil Moisture Content, m³/m³',
        ru='Объемное содержание влаги в почве, м³/м³',
    )),
    'SOILW_0.1-0.4m': Cfs2Band(5, 567, np.max, VarInfo(
        en='',
        ru='',
    )),
    'SOILW_0.4-1m': Cfs2Band(24, 617, np.max, VarInfo(
        en='',
        ru='',
    )),
    'SOILW_1-2m': Cfs2Band(25, 618, np.max, VarInfo(
        en='',
        ru='',
    )),
    'SPFH': Cfs2Band(18, 599, np.mean, VarInfo(
        en='Specific Humidity, kg/kg',
        ru='Удельная влажность, кг/кг',
    )),
    'TMAX': Cfs2Band(20, 601, np.max, VarInfo(
        en='Maximum Temperature, °C',
        ru='Максимальная температура, °C',
    )),
    'TMIN': Cfs2Band(21, 601, np.min, VarInfo(
        en='Minimum Temperature, °C',
        ru='Минимальная температура, °C',
    )),
    'TMP': Cfs2Band(17, 598, np.mean, VarInfo(
        en='Temperature, °C',
        ru='Температура, °C',
    )),
    'TMP_0m': Cfs2Band(3, 565, np.mean, VarInfo(
        en='Soil temperature, °C',
        ru='Температура почвы, °C',
    )),
    'TMP_0_0.1m': Cfs2Band(6, 568, np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_0.1_0.4m': Cfs2Band(7, 569, np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_0.4_1m': Cfs2Band(26, 619, np.mean, VarInfo(
        en='',
        ru='',
    )),
    'TMP_1-2m': Cfs2Band(27, 620, np.mean, VarInfo(
        en='',
        ru='',
    )),
    'UGRD': Cfs2Band(15, 596, np.mean, VarInfo(
        en='U-component of Wind, m/s',
        ru='Зональная составляющая ветра, т.е. компонент ветра по оси запад-восток, м/с',
    )),
    'ULWRF': Cfs2Band(10, 572, np.mean, VarInfo(
        en='Upward Long-Wave Radiation Flux, W/m²',
        ru='Восходящий поток инфракрасного излучения, Вт/м²',
    )),
    'USWRF': Cfs2Band(11, 575, np.mean, VarInfo(
        en='Upward Short-Wave Radiation Flux, W/m²',
        ru='Восходящий поток коротковолнового солнечного излучения, Вт/м²',
    )),
    'VGRD': Cfs2Band(16, 597, np.mean, VarInfo(
        en='V-component of Wind, m/s',
        ru='Меридиональная составляющая ветра, т.е. компонент ветра по оси юг-север, м/с',
    )),
    'WEASD': Cfs2Band(8, 570, np.sum, VarInfo(
        en='Water Equivalent of Accumulated Snow Depth, kg/m²',
        ru='Водный эквивалент накопленной глубины снега, кг/м²',
    )),
}
# fmt: on

CFS2_PGB_RESOLUTION = 1.0, 1.0
CFS2_PGB_BBOX = BoundingBox(-180.5, -90.5, 179.5, 90.5)

CFS2_PGB_PARAMS = {
    'var_RH': 'on',
    'lev_2_m_above_ground': 'on',
}

# fmt: off
CFS2_PGB_BANDS = {
    'RH': Cfs2Band(1, 368, np.mean, VarInfo(
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
CMIP6_RESOLUTION = 0.25
CMIP6_BBOX = BoundingBox(0.125, -59.875, 359.875, 89.875)

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
