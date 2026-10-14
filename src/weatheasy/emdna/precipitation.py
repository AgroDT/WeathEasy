from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from ._base import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, EMDNABase


if TYPE_CHECKING:
    from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


_PRECIP_KEYWORDS = ('precip', 'prec', 'rain', 'pcp', 'precipitation')
_TEMP_KEYWORDS = ('temperature', 'temp', 'tair')


class PrecipitationProcessor(EMDNABase):
    """EMDNA processor for daily precipitation.

    Uses a two-stage approach following Tang et al. (2021):
    1. Probability of Precipitation (PoP) via locally weighted average.
    2. Precipitation amount conditioned on PoP via spatiotemporally correlated
       random fields (SCRF) with cross-correlation to temperature range.
    """

    _OUTPUT_PREFIX: ClassVar[str] = 'emdna_precip_'
    _SPATIAL_CORR_KM: ClassVar[float] = 120.0
    _CROSS_CORR: ClassVar[float] = 0.3
    _PRECIP_THRESHOLD: ClassVar[float] = 0.1

    def _prepare_stations(self) -> None:
        available = self.stations.columns.tolist()

        precip_col = next(
            (c for c in available if any(k in c.lower() for k in _PRECIP_KEYWORDS)),
            None,
        )
        if precip_col is None:
            msg = 'Precipitation column not found in station observations'
            raise ValueError(msg)

        temp_col = next(
            (c for c in available if any(k in c.lower() for k in _TEMP_KEYWORDS)),
            None,
        )

        base_cols = ['station_id', 'lat', 'lon', 'date', 'year', 'month', 'day', precip_col]
        if temp_col:
            base_cols.append(temp_col)
        extra_cols = [
            c
            for c in available
            if any(
                k in c.lower()
                for k in ('elevation', 'height', 'alt', 'fabdem', 'dem', 'slope', 'gradient')
            )
        ]
        use_cols = [c for c in base_cols + extra_cols if c in available]

        observations = self.stations.loc[:, use_cols].copy()
        observations = observations.rename(columns={precip_col: 'precipitation'})
        if temp_col:
            observations = observations.rename(columns={temp_col: 'temperature'})
        observations['date'] = pd.to_datetime(
            observations['date']
            if 'date' in observations
            else observations[['year', 'month', 'day']],
            errors='coerce',
        )

        mask = (
            observations['lat'].between(LAT_MIN, LAT_MAX)
            & observations['lon'].between(LON_MIN, LON_MAX)
            & observations['date'].notna()
            & observations['precipitation'].notna()
            & (observations['precipitation'] >= 0)
        )
        self.stations = observations[mask].copy()
        self.elev_col, self.slope_we_col, self.slope_sn_col = self._get_predictor_cols(available)

        _LOG.info(
            'Loaded %d records from %d stations (%.1f%% dry days)',
            len(self.stations),
            self.stations['station_id'].nunique(),
            (self.stations['precipitation'] <= self._PRECIP_THRESHOLD).mean() * 100,
        )

    def _build_climatology(self) -> None:
        _LOG.info('Building precipitation climatology')
        self._station_clim: dict = {}
        for sid, grp in self.stations.groupby('station_id'):
            month_grp = grp.groupby(grp['date'].dt.month)

            monthly_mean = month_grp['precipitation'].mean()
            monthly_pop = month_grp['precipitation'].apply(
                lambda x: (x > self._PRECIP_THRESHOLD).mean(),
            )

            gamma_params: dict[int, dict[str, float]] = {}
            for m in range(1, 13):
                wet = grp.loc[grp['date'].dt.month == m, 'precipitation']
                wet = wet[wet > self._PRECIP_THRESHOLD]
                if len(wet) > 10:
                    try:
                        shape, _, scale = gamma.fit(wet, floc=0)
                        gamma_params[m] = {'shape': float(shape), 'scale': float(scale)}
                    except Exception:
                        gamma_params[m] = {'shape': 1.0, 'scale': float(wet.mean())}
                else:
                    gamma_params[m] = {'shape': 1.0, 'scale': 1.0}

            self._station_clim[sid] = {
                'monthly_mean': monthly_mean,
                'monthly_pop': monthly_pop,
                'gamma_params': gamma_params,
                'lat': float(grp['lat'].iloc[0]),
                'lon': float(grp['lon'].iloc[0]),
            }

    def _interpolate_climatology_to_grid(self) -> None:
        _LOG.info('Interpolating precipitation climatology to grid')
        self._grid_clim: dict[int, NDArray[np.float32]] = {}
        self._grid_pop_clim: dict[int, NDArray[np.float32]] = {}

        for m in range(1, 13):
            coords, means, pops = [], [], []
            for clim in self._station_clim.values():
                if m in clim['monthly_mean'].index:
                    coords.append([clim['lat'], clim['lon']])
                    means.append(clim['monthly_mean'][m])
                    pops.append(clim['monthly_pop'][m])

            if len(coords) > 3:
                arr = np.array(coords)
                self._grid_clim[m] = self._grid_interp(arr, np.array(means, dtype=float))
                pop_grid = self._grid_interp(arr, np.array(pops, dtype=float))
                self._grid_pop_clim[m] = np.clip(pop_grid, 0.0, 1.0)
            else:
                fill_mean = float(np.nanmean(means)) if means else 1.0
                fill_pop = float(np.nanmean(pops)) if pops else 0.2
                self._grid_clim[m] = np.full(self.grid_lats.shape, fill_mean, dtype=np.float32)
                self._grid_pop_clim[m] = np.full(self.grid_lats.shape, fill_pop, dtype=np.float32)

    def _cai(
        self,
        day_stations: pd.DataFrame,
        date_str: str,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Return (precipitation, pop, climatology, precip_uncertainty, pop_uncertainty)."""
        month = pd.Timestamp(date_str).month
        grid_clim = self._grid_clim[month]

        anomalies, pop_obs, coords, predictors = [], [], [], []

        for _, row in day_stations.iterrows():
            sid = row['station_id']
            clim = self._station_clim.get(sid)

            if clim is not None and month in clim['monthly_mean'].index:
                clim_val = float(clim['monthly_mean'][month])
            else:
                lat_i = int(np.argmin(np.abs(self.lats - row['lat'])))
                lon_i = int(np.argmin(np.abs(self.lons - row['lon'])))
                clim_val = float(grid_clim[lat_i, lon_i])

            anomalies.append(float(row['precipitation']) - clim_val)
            pop_obs.append(1.0 if float(row['precipitation']) > self._PRECIP_THRESHOLD else 0.0)
            coords.append([float(row['lat']), float(row['lon'])])
            predictors.append(
                self._row_predictors(row, self.elev_col, self.slope_we_col, self.slope_sn_col)
            )

        coords_arr = np.array(coords)
        interp, precip_unc = self._interpolate_anomalies(
            coords_arr, np.array(anomalies), np.array(predictors), min_std=0.5
        )
        pop, pop_unc = self._interpolate_weighted_avg(
            coords_arr, np.array(pop_obs), clip=(0.01, 0.99)
        )

        precip = np.maximum(grid_clim + interp, 0.0)
        return precip, pop, grid_clim, precip_unc, pop_unc

    def _load_era5(
        self, date_str: str
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
        if self.era5 is None:
            return None, None
        try:
            # Missing ERA5 days intentionally use stations; retain the broad fallback.
            ds = self.era5
            ts = pd.Timestamp(date_str)
            if 'tp' in ds.variables:
                era5_raw = ds.sel(date=ts)['tp'] * 1000.0
            elif 'precipitation' in ds.variables:
                era5_raw = ds.sel(date=ts)['precipitation']
            else:
                return None, None
            regridded = era5_raw.interp(longitude=self.lons, latitude=self.lats, method='linear')
            precip = np.maximum(regridded.to_numpy().astype(np.float32), 0.0)
            pop = np.where(np.isnan(precip), np.nan, precip > self._PRECIP_THRESHOLD).astype(
                np.float32
            )
        except Exception as exc:
            _LOG.warning('ERA5 load failed for %s: %s', date_str, exc)
            return None, None
        else:
            return precip, pop

    def _generate_ensemble(
        self,
        deterministic: NDArray[np.float32],
        pop_field: NDArray[np.float32],
        uncertainty: NDArray[np.float32],
        trange_scrf: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        shape = deterministic.shape
        ensemble = np.zeros((self.n_ensemble, *shape), dtype=np.float32)
        cross = self._CROSS_CORR

        for i in range(self.n_ensemble):
            spatial = self._spatial_random_field(shape)

            if trange_scrf is not None and trange_scrf.shape == shape and i in self._previous_scrf:
                scrf = cross * trange_scrf + np.sqrt(1.0 - cross**2) * self._previous_scrf[i]
            elif trange_scrf is not None and trange_scrf.shape == shape:
                scrf = cross * trange_scrf + np.sqrt(1.0 - cross**2) * spatial
            else:
                scrf = spatial

            self._previous_scrf[i] = scrf.copy()

            cum_prob = norm.cdf(scrf).astype(np.float32)
            precip_occurs = cum_prob > (1.0 - pop_field)

            member = np.zeros(shape, dtype=np.float32)
            p0 = 1.0 - pop_field
            valid = precip_occurs & (deterministic > 0) & (cum_prob > p0)

            if np.any(valid):
                denom = 1.0 - p0[valid] + 1e-8
                scaled = np.clip((cum_prob[valid] - p0[valid]) / denom, 1e-6, 1.0 - 1e-6)
                z = norm.ppf(scaled).astype(np.float32)
                member[valid] = np.maximum(deterministic[valid] + z * uncertainty[valid], 0.0)

            ensemble[i] = member

        return ensemble

    def _process_day(self, date_str: str, trange_scrf: NDArray[np.float32] | None = None) -> dict:
        out_file = f'emdna_precip_{date_str.replace("-", "")}.npz'
        if self.output_dir.exists(out_file):
            _LOG.info('%s already exists, skipping', out_file)
            return {
                'status': 'exists',
                'file': self.output_dir.fs.unstrip_protocol(self.output_dir.child(out_file).root),
            }

        target_date = pd.Timestamp(date_str)
        day_stations = self.stations[self.stations['date'] == target_date]

        if len(day_stations) < self.min_stations:
            _LOG.warning('Insufficient stations for %s: %d', date_str, len(day_stations))
            return {'status': 'insufficient_stations', 'count': len(day_stations)}

        n_wet = int((day_stations['precipitation'] > self._PRECIP_THRESHOLD).sum())
        _LOG.info('Processing %s (%d stations, %d with precip)', date_str, len(day_stations), n_wet)

        try:
            cai_precip, pop, climatology, precip_unc, pop_unc = self._cai(day_stations, date_str)

            era5_precip, era5_pop = self._load_era5(date_str)
            if era5_precip is not None and era5_pop is not None:
                final: NDArray[np.float32] = np.maximum(
                    0.75 * cai_precip + 0.25 * era5_precip, 0.0
                ).astype(np.float32)
                final = np.where(np.isnan(cai_precip), era5_precip, final).astype(np.float32)
                final = np.where(np.isnan(era5_precip), cai_precip, final).astype(np.float32)
                final_pop: NDArray[np.float32] = np.clip(
                    0.75 * pop + 0.25 * era5_pop, 0.0, 1.0
                ).astype(np.float32)
                final_pop = np.where(np.isnan(era5_pop), pop, final_pop).astype(np.float32)
            else:
                final = cai_precip
                final_pop = pop

            ensemble = self._generate_ensemble(final, final_pop, precip_unc, trange_scrf)
            e_mean = ensemble.mean(axis=0).astype(np.float32)
            e_std = ensemble.std(axis=0).astype(np.float32)
            e_pop = (ensemble > self._PRECIP_THRESHOLD).mean(axis=0).astype(np.float32)

            payload: dict = {
                'precipitation': final,
                'precipitation_cai': cai_precip,
                'pop': final_pop,
                'climatology': climatology,
                'precip_uncertainty': precip_unc,
                'pop_uncertainty': pop_unc,
                'ensemble': ensemble,
                'ensemble_mean': e_mean,
                'ensemble_std': e_std,
                'ensemble_pop': e_pop,
                'lats': self.lats,
                'lons': self.lons,
                'date': date_str,
                'n_stations': len(day_stations),
                'n_stations_wet': n_wet,
                'n_ensemble': self.n_ensemble,
                'precip_threshold': self._PRECIP_THRESHOLD,
                'spatial_corr_km': self._SPATIAL_CORR_KM,
                'cross_corr': self._CROSS_CORR,
                'elev_col': self.elev_col or 'none',
                'slope_we_col': self.slope_we_col or 'none',
                'slope_sn_col': self.slope_sn_col or 'none',
            }
            if era5_precip is not None:
                payload['era5_precipitation'] = era5_precip
            self._save_result(out_file, payload)
            _LOG.info('Saved %s', out_file)

            return {
                'status': 'success',
                'file': self.output_dir.fs.unstrip_protocol(self.output_dir.child(out_file).root),
                'n_stations': len(day_stations),
                'n_stations_wet': n_wet,
                'precip_range': (float(np.nanmin(final)), float(np.nanmax(final))),
                'pop_range': (float(np.nanmin(final_pop)), float(np.nanmax(final_pop))),
            }

        except Exception as exc:
            _LOG.error('Failed for %s: %s', date_str, exc, exc_info=True)
            return {'status': 'error', 'error': str(exc)}
