"""Temperature bias corrector (inference).

Port of the production ``BiasCorrector`` adapted to read a daily grid from the Zarr
store via :class:`~weatheasy.correction.sources.Source` instead of a NetCDF file.
A single XGBoost regressor predicts the ERA5/CFS2 bias from terrain + seasonal-bias
features; the prediction is suppressed where the seasonal bias is small.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from . import features
from .climatology import TEMP_FILES, Climatology, verify_manifest


if TYPE_CHECKING:
    from datetime import date

    import zarr
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)

FEATURE_COLS = [
    'lat',
    'lon',
    'elevation',
    'slope_we',
    'slope_sn',
    'era5_seasonal_bias',
    'month',
    'doy_sin',
    'doy_cos',
    'era5_value',
    'era5_anomaly',
]
CORRECTION_THRESHOLD = 0.3  # deg C; below this seasonal bias we leave the value untouched


class TemperatureCorrector:
    """Load per-source artifacts and bias-correct a daily temperature grid."""

    def __init__(self, source: Source, artifacts: FileStore, dem_store: Store) -> None:
        self.source = source
        self.dir = artifacts.child(source.name, 'temp')
        self.dem_store = dem_store
        self.model: Any = None
        self.climatology: Climatology | None = None
        self.seasonal_bias: dict[int, NDArray[np.float32]] | None = None
        self.grid_lats: NDArray[np.float32] | None = None
        self.grid_lons: NDArray[np.float32] | None = None
        self._dem_cache: dict[tuple[int, int], features.Dem] = {}

    def load_artifacts(self) -> TemperatureCorrector:
        verify_manifest(self.dir, TEMP_FILES)
        self.climatology = Climatology.load(self.dir, self.source, self.source.temp_var)
        with self.dir.open('model.joblib') as stream:
            self.model = joblib.load(stream)
        with (
            self.dir.open('seasonal_bias_grid.npz') as stream,
            np.load(stream, allow_pickle=False) as data,
        ):
            self.seasonal_bias = {m: data[f'month_{m}'] for m in range(1, 13)}
        with self.dir.open('lats.npy') as stream:
            self.grid_lats = np.load(stream, allow_pickle=False)
        with self.dir.open('lons.npy') as stream:
            self.grid_lons = np.load(stream, allow_pickle=False)
        _LOG.info('Loaded %s temperature artifacts from %s', self.source.name, self.dir.root)
        return self

    @property
    def artifact_id(self) -> str:
        if self.climatology is None:
            msg = 'Temperature artifacts not loaded'
            raise RuntimeError(msg)
        return str(self.climatology.metadata['training_id'])

    def correct(self, root: zarr.Group, day: date) -> dict[str, NDArray[np.float32]]:
        if (
            self.model is None
            or self.climatology is None
            or self.seasonal_bias is None
            or self.grid_lats is None
            or self.grid_lons is None
        ):
            msg = 'Artifacts not loaded; call load_artifacts() first'
            raise RuntimeError(msg)

        grid, lats, lons = self.source.read_day(root, self.source.temp_var, day)
        self.climatology.require_grid(lats, lons)
        monthly_mean = self.climatology.mean[day.month - 1]
        anomaly = (grid - monthly_mean).astype(np.float32)
        elev, slope_we, slope_sn = self._dem_arrays(lats, lons)
        seasonal = features.seasonal_field(
            self.seasonal_bias,
            self.grid_lats,
            self.grid_lons,
            day.month,
            lats,
            lons,
        )
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        doy = day.timetuple().tm_yday
        feat = pd.DataFrame(
            {
                'lat': lat_grid.ravel(),
                'lon': lon_grid.ravel(),
                'elevation': elev,
                'slope_we': slope_we,
                'slope_sn': slope_sn,
                'era5_seasonal_bias': seasonal.ravel(),
                'month': np.int8(day.month),
                'doy_sin': np.float32(np.sin(2 * np.pi * doy / 365)),
                'doy_cos': np.float32(np.cos(2 * np.pi * doy / 365)),
                'era5_value': grid.ravel(),
                'era5_anomaly': anomaly.ravel(),
            }
        )
        valid = np.isfinite(grid) & np.isfinite(monthly_mean)
        bias = np.full(grid.shape, np.nan, np.float32)
        if valid.any():
            predicted = self.model.predict(
                feat.iloc[np.flatnonzero(valid.ravel())][FEATURE_COLS]
                .astype(np.float32)
                .to_numpy(),
            )
            bias[valid] = predicted.astype(np.float32)
        bias[(np.abs(seasonal) < CORRECTION_THRESHOLD) & valid] = 0.0
        corrected = (grid - bias).astype(np.float32)
        _LOG.info(
            '%s %s: mean %.2f -> %.2f degC (corrected %.0f%% of cells)',
            self.source.name,
            day,
            float(grid.mean()),
            float(corrected.mean()),
            float((bias != 0).mean() * 100),
        )
        return {'original': grid, 'corrected': corrected, 'bias': bias, 'lats': lats, 'lons': lons}

    def _dem_arrays(
        self,
        lats: NDArray,
        lons: NDArray,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        key = (len(lats), len(lons))
        if key not in self._dem_cache:
            dem = features.load_dem(self.dem_store)
            self._dem_cache[key] = features.resample_dem(dem, lats, lons)
        return features.flat_dem(self._dem_cache[key])
