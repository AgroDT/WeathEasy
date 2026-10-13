"""Precipitation bias corrector (inference).

Port of the production two-stage ``PrecipCorrector`` (XGBoost wet/dry classifier +
amount regressor) adapted to read a daily grid from the Zarr store. Hybrid spatial
scheme: cells within ``HYBRID_RADIUS_KM`` of a training station use the full model
with seasonal fields; far cells use a fallback model trained without them. The grid
wet threshold is calibrated so the wet-cell fraction matches the source climatology.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from . import features
from .climatology import PRECIP_FILES, Climatology, verify_manifest


if TYPE_CHECKING:
    from datetime import date

    import zarr
    from numpy.typing import NDArray

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)

_BASE_FEAT = [
    'lat',
    'lon',
    'elevation',
    'slope_we',
    'slope_sn',
    'log_dist_coast',
    'month',
    'doy_sin',
    'doy_cos',
    'era5_precip',
    'era5_precip_log',
    'era5_precip_anom',
    'era5_wet_fraction',
]
_SEASONAL_FEAT = ['era5_seasonal_log_ratio', 'precip_wet_bias']
WET_FEAT = _BASE_FEAT + _SEASONAL_FEAT
REG_FEAT = WET_FEAT

WET_THRESHOLD = 1.0  # mm/day
HYBRID_RADIUS_KM = 200.0
WET_BIAS_THRESHOLD = 0.05  # below this seasonal wet bias we leave the value untouched
_WET_FRAC_CALIB = 0.65  # the source overestimates wet fraction
_CALIB_SAMPLE = 50_000


class PrecipCorrector:
    """Load per-source artifacts and bias-correct a daily precipitation grid (mm/day)."""

    def __init__(self, source: Source, artifacts: FileStore, terrain: features.Dem | None) -> None:
        self.source = source
        self.dir = artifacts.child(source.name, 'precip')
        self.terrain = terrain
        self.clf: Any = None
        self.climatology: Climatology | None = None
        self.reg: Any = None
        self.fallback_clf: Any = None
        self.fallback_reg: Any = None
        self.threshold = 0.5
        self.fallback_threshold = 0.5
        self.seasonal: dict[str, dict[int, NDArray[np.float32]]] | None = None
        self.train_coords: NDArray[np.float32] | None = None
        self.grid_lats: NDArray[np.float32] | None = None
        self.grid_lons: NDArray[np.float32] | None = None

    def load_artifacts(self) -> PrecipCorrector:
        verify_manifest(self.dir, PRECIP_FILES)
        self.climatology = Climatology.load(self.dir, self.source, self.source.precip_var)
        with self.dir.open('clf.joblib') as stream:
            self.clf = joblib.load(stream)
        with self.dir.open('reg.joblib') as stream:
            self.reg = joblib.load(stream)
        with self.dir.open('fallback_clf.joblib') as stream:
            self.fallback_clf = joblib.load(stream)
        with self.dir.open('fallback_reg.joblib') as stream:
            self.fallback_reg = joblib.load(stream)
        with self.dir.open('train_coords.npy') as stream:
            self.train_coords = np.load(stream, allow_pickle=False)
        with self.dir.open('lats.npy') as stream:
            self.grid_lats = np.load(stream, allow_pickle=False)
        with self.dir.open('lons.npy') as stream:
            self.grid_lons = np.load(stream, allow_pickle=False)
        with self.dir.open('seasonal.npz') as stream, np.load(stream, allow_pickle=False) as data:
            self.threshold = float(data['threshold'])
            self.fallback_threshold = float(data['fallback_threshold'])
            self.seasonal = {
                'log_ratio': {m: data[f'log_ratio_m{m}'] for m in range(1, 13)},
                'wet_bias': {m: data[f'wet_bias_m{m}'] for m in range(1, 13)},
            }
        _LOG.info('Loaded %s precip artifacts from %s', self.source.name, self.dir.root)
        return self

    @property
    def artifact_id(self) -> str:
        if self.climatology is None:
            msg = 'Precipitation artifacts not loaded'
            raise RuntimeError(msg)
        return str(self.climatology.metadata['training_id'])

    def correct(self, root: zarr.Group, day: date) -> dict[str, NDArray[np.float32]]:
        data = self.source.read_day(root, self.source.precip_var, day)
        return self.correct_grid(day, *data)

    def correct_grid(
        self,
        day: date,
        raw: NDArray[np.float32],
        lats: NDArray[np.float32],
        lons: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        if (
            self.clf is None
            or self.climatology is None
            or self.seasonal is None
            or self.train_coords is None
            or self.grid_lats is None
            or self.grid_lons is None
        ):
            msg = 'Artifacts not loaded; call load_artifacts() first'
            raise RuntimeError(msg)

        self.climatology.require_grid(lats, lons)
        era5_day = np.maximum(raw * self.source.precip_to_mm, 0.0).astype(np.float32)
        valid = np.isfinite(era5_day)
        if not valid.any():
            return {'original': era5_day, 'corrected': era5_day.copy(), 'lats': lats, 'lons': lons}
        n_lat, n_lon = era5_day.shape
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        flat_lats, flat_lons = lat_grid.ravel(), lon_grid.ravel()

        if self.terrain is None:
            msg = 'Prepared terrain has not been loaded'
            raise RuntimeError(msg)
        elev, slope_we, slope_sn = features.flat_dem(self.terrain)
        coast = np.log1p(features.dist_to_coast(flat_lats, flat_lons)).astype(np.float32)
        monthly_mean = self.climatology.mean[day.month - 1]
        wet_fraction = self.climatology.require_wet_fraction()[day.month - 1]
        anomaly = (era5_day - monthly_mean).astype(np.float32)
        # The daily wet fraction still calibrates the day's output count; the
        # model feature below is the frozen cell/month temporal fraction.
        predictable = valid & np.isfinite(monthly_mean) & np.isfinite(wet_fraction)
        target_wet = (
            float((era5_day[predictable] >= WET_THRESHOLD).mean()) * _WET_FRAC_CALIB
            if predictable.any()
            else 0.0
        )
        log_ratio = features.seasonal_field(
            self.seasonal['log_ratio'],
            self.grid_lats,
            self.grid_lons,
            day.month,
            lats,
            lons,
        )
        wet_bias = features.seasonal_field(
            self.seasonal['wet_bias'],
            self.grid_lats,
            self.grid_lons,
            day.month,
            lats,
            lons,
        )

        feat = _build_features(
            flat_lats,
            flat_lons,
            elev,
            slope_we,
            slope_sn,
            coast,
            day,
            era5_day.ravel(),
            anomaly.ravel(),
            wet_fraction.ravel(),
            log_ratio.ravel(),
            wet_bias.ravel(),
        )

        dists = features.point_distances_km(
            np.column_stack([flat_lats, flat_lons]),
            self.train_coords,
        )
        near = dists.min(axis=1) < HYBRID_RADIUS_KM
        threshold = (
            self._calibrate(feat.iloc[np.flatnonzero(predictable)], target_wet)
            if predictable.any()
            else self.threshold
        )

        corrected = np.full(n_lat * n_lon, np.nan, np.float32)
        near_idx = np.where(near & predictable.ravel())[0]
        if near_idx.size:
            corrected[near_idx] = _predict(feat.iloc[near_idx], self.clf, self.reg, threshold)
        far_idx = np.where(~near & predictable.ravel())[0]
        if far_idx.size:
            far_feat = feat.iloc[far_idx].copy()
            far_feat[_SEASONAL_FEAT] = np.float32(0.0)
            corrected[far_idx] = _predict(
                far_feat,
                self.fallback_clf,
                self.fallback_reg,
                self.fallback_threshold,
            )

        suppress = (np.abs(wet_bias.ravel()) < WET_BIAS_THRESHOLD) & predictable.ravel()
        corrected[suppress] = era5_day.ravel()[suppress]
        corrected_grid = np.maximum(corrected.reshape(n_lat, n_lon), 0.0).astype(np.float32)

        _LOG.info(
            '%s %s: precip mean %.2f -> %.2f mm (near %.0f%%)',
            self.source.name,
            day,
            float(era5_day.mean()),
            float(corrected_grid.mean()),
            float(near.mean() * 100),
        )
        return {'original': era5_day, 'corrected': corrected_grid, 'lats': lats, 'lons': lons}

    def _calibrate(self, feat: pd.DataFrame, target: float) -> float:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(feat), min(_CALIB_SAMPLE, len(feat)), replace=False)
        probs = self.clf.predict_proba(
            feat.iloc[sample][WET_FEAT].astype(np.float32).to_numpy(),
        )[:, 1]
        best_t, best_score = self.threshold, np.inf
        for t in np.linspace(0.05, 0.95, 91):
            score = abs(float((probs >= t).mean()) - target)
            if score < best_score:
                best_score, best_t = score, float(t)
        return best_t


def _build_features(  # noqa: PLR0917
    flat_lats: NDArray,
    flat_lons: NDArray,
    elev: NDArray,
    slope_we: NDArray,
    slope_sn: NDArray,
    coast: NDArray,
    day: date,
    era5_flat: NDArray,
    anom_flat: NDArray,
    wet_frac_flat: NDArray,
    log_ratio_flat: NDArray,
    wet_bias_flat: NDArray,
) -> pd.DataFrame:
    doy = day.timetuple().tm_yday
    return pd.DataFrame(
        {
            'lat': flat_lats,
            'lon': flat_lons,
            'elevation': elev,
            'slope_we': slope_we,
            'slope_sn': slope_sn,
            'log_dist_coast': coast,
            'month': np.int8(day.month),
            'doy_sin': np.float32(np.sin(2 * np.pi * doy / 365)),
            'doy_cos': np.float32(np.cos(2 * np.pi * doy / 365)),
            'era5_precip': era5_flat,
            'era5_precip_log': np.log1p(era5_flat).astype(np.float32),
            'era5_precip_anom': anom_flat,
            'era5_wet_fraction': wet_frac_flat,
            'era5_seasonal_log_ratio': log_ratio_flat,
            'precip_wet_bias': wet_bias_flat,
        }
    )


def _predict(feat: pd.DataFrame, clf: Any, reg: Any, threshold: float) -> NDArray[np.float32]:
    """Two-stage prediction: classify wet/dry, then regress amount (mm) on wet cells."""
    probs = clf.predict_proba(feat[WET_FEAT].astype(np.float32).to_numpy())[:, 1]
    result = np.zeros(len(feat), np.float32)
    wet = np.where(probs >= threshold)[0]
    if wet.size:
        amount = reg.predict(feat.iloc[wet][REG_FEAT].astype(np.float32).to_numpy())
        result[wet] = np.expm1(np.maximum(amount, 0.0))
    return result
