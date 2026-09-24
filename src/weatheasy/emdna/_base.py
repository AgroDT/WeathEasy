from __future__ import annotations

import copy
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


if TYPE_CHECKING:
    from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)

LON_MIN: float = 19.0
LON_MAX: float = 180.0
LAT_MIN: float = 41.0
LAT_MAX: float = 82.0
RESOLUTION: float = 0.1
KM_PER_DEG: float = 111.0


class EMDNABase(ABC):
    stations: pd.DataFrame
    _OUTPUT_PREFIX: ClassVar[str]
    _SPATIAL_CORR_KM: ClassVar[float] = 150.0

    def __init__(
        self,
        station_file: str | Path,
        output_dir: str | Path,
        *,
        era5_dir: str | Path | None = None,
        max_stations: int = 30,
        min_stations: int = 20,
        search_radius_km: float = 400.0,
        n_ensemble: int = 100,
    ) -> None:
        self.station_file = Path(station_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.era5_dir = Path(era5_dir) if era5_dir else None

        if not 1 <= min_stations <= max_stations or n_ensemble < 1 or search_radius_km <= 0:
            msg = 'Require positive ensemble/radius and 1 <= min_stations <= max_stations'
            raise ValueError(msg)

        self.max_stations = max_stations
        self.min_stations = min_stations
        self.search_radius_km = search_radius_km
        self.n_ensemble = n_ensemble

        self._previous_scrf: dict[int, NDArray[np.float32]] = {}
        self._rng = np.random.default_rng()
        self._last_date: date | None = None

        self._setup_grid()
        self._load_stations()
        self._build_climatology()
        self._interpolate_climatology_to_grid()

    # --- Grid ---

    def _setup_grid(self) -> None:
        self.lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION, dtype=np.float32)
        self.lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION, dtype=np.float32)
        self.grid_lons, self.grid_lats = np.meshgrid(self.lons, self.lats)
        _LOG.info('Grid: %d x %d', len(self.lats), len(self.lons))

    # --- Abstract methods ---

    @abstractmethod
    def _load_stations(self) -> None: ...

    @abstractmethod
    def _build_climatology(self) -> None: ...

    @abstractmethod
    def _interpolate_climatology_to_grid(self) -> None: ...

    @abstractmethod
    def _process_day(self, date_str: str) -> dict: ...

    # --- Shared interpolation helpers ---

    def _grid_interp(
        self,
        station_coords: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        fill = float(np.nanmean(values))
        pts = np.column_stack([self.grid_lats.ravel(), self.grid_lons.ravel()])
        out = griddata(station_coords, values, pts, method='linear', fill_value=fill)
        return out.reshape(self.grid_lats.shape).astype(np.float32)

    def _select_stations(
        self,
        target: NDArray[np.float64],
        coords: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        dist = self._distances(target, coords)
        mask = dist <= self.search_radius_km
        n = int(mask.sum())
        if n < self.min_stations:
            idx = np.argsort(dist)[: self.min_stations]
            mask = np.zeros(len(dist), dtype=bool)
            mask[idx] = True
        elif n > self.max_stations:
            within = np.where(mask)[0]
            top = within[np.argsort(dist[within])[: self.max_stations]]
            mask = np.zeros(len(dist), dtype=bool)
            mask[top] = True
        return mask

    def _tricubic_weights(self, dist: NDArray[np.float64]) -> NDArray[np.float64]:
        u = dist / (dist.max() + 1.0)
        return np.where(u <= 1.0, (1.0 - u**3) ** 3, 0.0)

    def _weighted_regression(
        self,
        design: NDArray[np.float64],
        y: NDArray[np.float64],
        w: NDArray[np.float64],
        x_pred: NDArray[np.float64],
        min_std: float = 0.1,
    ) -> tuple[float, float]:
        weights = np.diag(w)
        weighted_design = design.T @ weights
        try:
            inverse = np.linalg.inv(weighted_design @ design)
        except np.linalg.LinAlgError:
            pred = float(np.average(y, weights=w))
            std = float(np.sqrt(np.average((y - pred) ** 2, weights=w)))
            return pred, max(std, min_std)
        beta = inverse @ weighted_design @ y
        pred = float(x_pred @ beta)
        res = y - design @ beta
        mse = float((w * res**2).sum() / w.sum())
        var = mse * float(x_pred @ inverse @ x_pred)
        return pred, max(float(np.sqrt(max(var, 0.0))), min_std)

    def _interpolate_anomalies(
        self,
        coords: NDArray[np.float64],
        anomalies: NDArray[np.float64],
        predictors: NDArray[np.float64],
        min_std: float = 0.1,
        chunk_size: int = 8000,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        target_pts = np.column_stack([self.grid_lats.ravel(), self.grid_lons.ravel()])
        n = len(target_pts)
        result = np.zeros(n, dtype=np.float32)
        uncertainty = np.full(n, 2.0, dtype=np.float32)

        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            for k, pt in enumerate(target_pts[i:j]):
                mask = self._select_stations(pt, coords)
                dist = self._distances(pt, coords)[mask]
                sel_y = anomalies[mask]
                sel_p = predictors[mask]

                if len(sel_y) < 3:
                    continue

                w = self._tricubic_weights(dist)
                if w.sum() == 0.0:
                    continue

                design = np.column_stack([np.ones(len(sel_p)), sel_p])
                x_pred = np.array([1.0, pt[0], pt[1], 0.0, 0.0, 0.0])
                pred, std = self._weighted_regression(design, sel_y, w, x_pred, min_std)
                result[i + k] = pred
                uncertainty[i + k] = std

        return (
            result.reshape(self.grid_lats.shape),
            uncertainty.reshape(self.grid_lats.shape),
        )

    def _interpolate_weighted_avg(
        self,
        coords: NDArray[np.float64],
        values: NDArray[np.float64],
        clip: tuple[float, float] | None = None,
        chunk_size: int = 8000,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Interpolate scalar values via locally weighted average; also returns uncertainty."""
        default = float(np.nanmean(values))
        target_pts = np.column_stack([self.grid_lats.ravel(), self.grid_lons.ravel()])
        n = len(target_pts)
        result = np.full(n, default, dtype=np.float32)
        uncertainty = np.full(n, 0.2, dtype=np.float32)

        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            for k, pt in enumerate(target_pts[i:j]):
                mask = self._select_stations(pt, coords)
                dist = self._distances(pt, coords)[mask]
                sel = values[mask]

                if len(sel) < 3:
                    continue

                if np.all(sel == sel[0]):
                    result[i + k] = sel[0]
                    uncertainty[i + k] = 0.05
                    continue

                w = self._tricubic_weights(dist)
                if w.sum() == 0.0:
                    continue

                pred = float(np.average(sel, weights=w))
                if clip is not None:
                    pred = float(np.clip(pred, clip[0], clip[1]))
                var = float(np.average((sel - pred) ** 2, weights=w))
                result[i + k] = pred
                uncertainty[i + k] = float(np.sqrt(max(var, 0.01)))

        return (
            result.reshape(self.grid_lats.shape),
            uncertainty.reshape(self.grid_lats.shape),
        )

    # --- Random field generation (SCRF) ---

    def _spatial_random_field(self, shape: tuple[int, int]) -> NDArray[np.float32]:
        sigma = self._SPATIAL_CORR_KM / KM_PER_DEG / RESOLUTION
        noise = self._rng.normal(0.0, 1.0, shape)
        field = gaussian_filter(noise, sigma=sigma)
        std = field.std()
        if std > 0:
            field = (field - field.mean()) / std
        return field.astype(np.float32)

    def _scrf(
        self, shape: tuple[int, int], temporal_corr: float, member: int
    ) -> NDArray[np.float32]:
        spatial = self._spatial_random_field(shape)
        if member in self._previous_scrf and self._previous_scrf[member].shape == spatial.shape:
            result = (
                temporal_corr * self._previous_scrf[member]
                + np.sqrt(1.0 - temporal_corr**2) * spatial
            )
        else:
            result = spatial
        self._previous_scrf[member] = result.copy()
        return result

    # --- Station predictor helpers ---

    def _get_predictor_cols(
        self, available: list[str]
    ) -> tuple[str | None, str | None, str | None]:
        elev_col = next(
            (
                c
                for c in available
                if any(k in c.lower() for k in ('elevation', 'height', 'alt', 'fabdem', 'dem'))
            ),
            None,
        )
        slope_we = next(
            (
                c
                for c in available
                if any(k in c.lower() for k in ('we', 'ew', 'east', 'west'))
                and 'slope' in c.lower()
            ),
            None,
        )
        slope_sn = next(
            (
                c
                for c in available
                if any(k in c.lower() for k in ('sn', 'ns', 'south', 'north'))
                and 'slope' in c.lower()
            ),
            None,
        )
        return elev_col, slope_we, slope_sn

    def _row_predictors(
        self,
        row: pd.Series,
        elev_col: str | None,
        slope_we_col: str | None,
        slope_sn_col: str | None,
    ) -> list[float]:
        def _val(col: str | None) -> float:
            if col is None:
                return 0.0
            v = row.get(col, np.nan)
            return 0.0 if pd.isna(v) else float(v)

        return [
            float(row['lat']),
            float(row['lon']),
            _val(elev_col),
            _val(slope_we_col),
            _val(slope_sn_col),
        ]

    # --- Period processing ---

    @staticmethod
    def _distances(target: NDArray, coords: NDArray) -> NDArray[np.float64]:
        delta_lat = coords[:, 0] - target[0]
        delta_lon = (coords[:, 1] - target[1] + 180.0) % 360.0 - 180.0
        return np.hypot(delta_lat, delta_lon * np.cos(np.deg2rad(target[0]))) * KM_PER_DEG

    def _save_result(self, path: Path, payload: dict[str, Any]) -> None:
        # Results and restart state become visible together, only after a complete write.
        payload['scrf_state'] = np.stack([self._previous_scrf[i] for i in range(self.n_ensemble)])
        payload['rng_state'] = json.dumps(self._rng.bit_generator.state)
        with tempfile.NamedTemporaryFile(
            dir=self.output_dir, suffix='.npz', delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(temporary, **payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_result(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as result:
            states = result['scrf_state']
            if (
                states.shape != (self.n_ensemble, *self.grid_lats.shape)
                or not np.array_equal(result['lats'], self.lats)
                or not np.array_equal(result['lons'], self.lons)
            ):
                msg = f'Incompatible ensemble or grid in {path}'
                raise ValueError(msg)
            self._previous_scrf = {i: states[i].copy() for i in range(self.n_ensemble)}
            self._rng.bit_generator.state = json.loads(str(result['rng_state']))
            self._last_date = date.fromisoformat(str(result['date']))

    def process_day(self, date_str: str, **kwargs: Any) -> dict:
        current = date.fromisoformat(date_str)
        path = self.output_dir / f'{self._OUTPUT_PREFIX}{current:%Y%m%d}.npz'
        if path.exists():
            self._restore_result(path)
            return {'status': 'exists', 'file': path}
        previous = sorted(self.output_dir.glob(f'{self._OUTPUT_PREFIX}????????.npz'))
        if previous:
            latest = previous[-1]
            latest_date = date.fromisoformat(latest.stem.removeprefix(self._OUTPUT_PREFIX))
            if current <= latest_date:
                msg = 'Cannot insert earlier dates into an existing ensemble sequence'
                raise ValueError(msg)
            if self._last_date != latest_date:
                self._restore_result(latest)
        states = {i: field.copy() for i, field in self._previous_scrf.items()}
        rng_state = copy.deepcopy(self._rng.bit_generator.state)
        result = self._process_day(date_str, **kwargs)
        if result['status'] == 'success':
            self._last_date = current
        else:
            self._previous_scrf = states
            self._rng.bit_generator.state = rng_state
        return result

    def process_period(self, start: date | None = None, end: date | None = None) -> list[dict]:
        if start is not None and end is not None and start > end:
            msg = 'Start date must not follow end date'
            raise ValueError(msg)
        dates = sorted(self.stations['date'].dt.date.unique())
        previous = sorted(self.output_dir.glob(f'{self._OUTPUT_PREFIX}????????.npz'))
        last = None
        if previous:
            self._restore_result(previous[-1])
            last = self._last_date
        results = []
        for day in dates:
            if (start is not None and day < start) or (end is not None and day > end):
                continue
            if last is not None and day <= last:
                continue
            result = self.process_day(day.isoformat())
            results.append(result)
            if result['status'] != 'success':
                # Do not advance past a failed/unfinished day; it must remain retryable.
                break
        return results
