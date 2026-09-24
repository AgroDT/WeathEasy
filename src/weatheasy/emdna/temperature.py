from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from ._base import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, EMDNABase


if TYPE_CHECKING:
    from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


class TemperatureProcessor(EMDNABase):
    """EMDNA processor for daily mean temperature.

    Implements Climatologically Aided Interpolation (CAI) with a 100-member
    spatiotemporally correlated ensemble following Tang et al. (2021).
    """

    _OUTPUT_PREFIX: ClassVar[str] = 'emdna_temp_'
    _SPATIAL_CORR_KM: ClassVar[float] = 150.0
    _TEMPORAL_CORR: ClassVar[float] = 0.8

    def _load_stations(self) -> None:
        _LOG.info('Loading stations from %s', self.station_file)
        sample = pd.read_csv(self.station_file, nrows=1)
        available = sample.columns.tolist()

        base_cols = ['station_id', 'lat', 'lon', 'year', 'month', 'day', 'temperature']
        extra_cols = [
            c
            for c in available
            if any(
                k in c.lower()
                for k in ('elevation', 'height', 'alt', 'fabdem', 'dem', 'slope', 'gradient')
            )
        ]
        use_cols = [c for c in base_cols + extra_cols if c in available]

        observations = pd.read_csv(self.station_file, usecols=use_cols)
        observations['date'] = pd.to_datetime(
            observations[['year', 'month', 'day']], errors='coerce'
        )

        mask = (
            observations['lat'].between(LAT_MIN, LAT_MAX)
            & observations['lon'].between(LON_MIN, LON_MAX)
            & observations['date'].notna()
            & observations['temperature'].notna()
        )
        self.stations = observations[mask].copy()
        self.elev_col, self.slope_we_col, self.slope_sn_col = self._get_predictor_cols(available)

        _LOG.info(
            'Loaded %d records from %d stations',
            len(self.stations),
            self.stations['station_id'].nunique(),
        )

    def _build_climatology(self) -> None:
        _LOG.info('Building temperature climatology')
        self._station_clim: dict = {}
        for sid, grp in self.stations.groupby('station_id'):
            self._station_clim[sid] = {
                'monthly': grp.groupby(grp['date'].dt.month)['temperature'].mean(),
                'lat': float(grp['lat'].iloc[0]),
                'lon': float(grp['lon'].iloc[0]),
            }

    def _interpolate_climatology_to_grid(self) -> None:
        _LOG.info('Interpolating temperature climatology to grid')
        self._grid_clim: dict[int, NDArray[np.float32]] = {}
        for m in range(1, 13):
            coords, vals = [], []
            for clim in self._station_clim.values():
                if m in clim['monthly'].index:
                    coords.append([clim['lat'], clim['lon']])
                    vals.append(clim['monthly'][m])
            if len(coords) > 3:
                self._grid_clim[m] = self._grid_interp(
                    np.array(coords), np.array(vals, dtype=float)
                )
            else:
                fill = float(np.nanmean(vals)) if vals else 0.0
                self._grid_clim[m] = np.full(self.grid_lats.shape, fill, dtype=np.float32)

    def _clim_value_at_station(self, station_id: object, month: int) -> float | None:
        clim = self._station_clim.get(station_id)
        if clim is None:
            return None
        monthly = clim['monthly']
        return float(monthly[month]) if month in monthly.index else None

    def _cai(
        self,
        day_stations: pd.DataFrame,
        date_str: str,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        month = pd.Timestamp(date_str).month
        grid_clim = self._grid_clim[month]

        anomalies, coords, predictors = [], [], []
        for _, row in day_stations.iterrows():
            clim_val = self._clim_value_at_station(row['station_id'], month)
            if clim_val is None:
                lat_i = int(np.argmin(np.abs(self.lats - row['lat'])))
                lon_i = int(np.argmin(np.abs(self.lons - row['lon'])))
                clim_val = float(grid_clim[lat_i, lon_i])
            anomalies.append(float(row['temperature']) - clim_val)
            coords.append([float(row['lat']), float(row['lon'])])
            predictors.append(
                self._row_predictors(row, self.elev_col, self.slope_we_col, self.slope_sn_col)
            )

        interp, unc = self._interpolate_anomalies(
            np.array(coords),
            np.array(anomalies),
            np.array(predictors),
            min_std=0.1,
        )
        return grid_clim + interp, unc

    def _load_era5(self, date_str: str) -> NDArray[np.float32] | None:
        if self.era5_dir is None:
            return None
        try:
            import xarray as xr

            files = sorted(self.era5_dir.glob(f'*{date_str[:7].replace("-", "")}*daily.nc'))
            if not files:
                _LOG.warning('No daily ERA5 file for %s; using stations', date_str)
                return None
            with xr.open_dataset(files[0]) as ds:
                day = ds.sel(date=pd.Timestamp(date_str))['t2m_mean']
                regridded = day.interp(longitude=self.lons, latitude=self.lats, method='linear')
                return regridded.to_numpy().astype(np.float32)
        except Exception as exc:
            _LOG.warning('ERA5 load failed for %s: %s', date_str, exc)
            return None

    def _generate_ensemble(
        self,
        deterministic: NDArray[np.float32],
        uncertainty: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        shape = deterministic.shape
        ensemble = np.empty((self.n_ensemble, *shape), dtype=np.float32)
        for i in range(self.n_ensemble):
            scrf = self._scrf(shape, self._TEMPORAL_CORR, i)
            ensemble[i] = deterministic + scrf * uncertainty
        return ensemble

    def _process_day(self, date_str: str) -> dict:
        out_file = self.output_dir / f'emdna_temp_{date_str.replace("-", "")}.npz'
        if out_file.exists():
            _LOG.info('%s already exists, skipping', out_file.name)
            return {'status': 'exists', 'file': out_file}

        target_date = pd.Timestamp(date_str)
        day_stations = self.stations[self.stations['date'] == target_date]

        if len(day_stations) < self.min_stations:
            _LOG.warning('Insufficient stations for %s: %d', date_str, len(day_stations))
            return {'status': 'insufficient_stations', 'count': len(day_stations)}

        _LOG.info('Processing %s (%d stations)', date_str, len(day_stations))

        try:
            cai_result, uncertainty = self._cai(day_stations, date_str)

            era5 = self._load_era5(date_str)
            if era5 is not None:
                final: NDArray[np.float32] = 0.7 * cai_result + 0.3 * era5
                final = np.where(np.isnan(cai_result), era5, final).astype(np.float32)
                final = np.where(np.isnan(era5), cai_result, final).astype(np.float32)
            else:
                final = cai_result

            ensemble = self._generate_ensemble(final, uncertainty)
            e_mean = ensemble.mean(axis=0).astype(np.float32)
            e_std = ensemble.std(axis=0).astype(np.float32)

            payload: dict = {
                'temperature': final,
                'temperature_cai': cai_result,
                'uncertainty': uncertainty,
                'ensemble': ensemble,
                'ensemble_mean': e_mean,
                'ensemble_std': e_std,
                'lats': self.lats,
                'lons': self.lons,
                'date': date_str,
                'n_stations': len(day_stations),
                'n_ensemble': self.n_ensemble,
                'spatial_corr_km': self._SPATIAL_CORR_KM,
                'temporal_corr': self._TEMPORAL_CORR,
                'elev_col': self.elev_col or 'none',
                'slope_we_col': self.slope_we_col or 'none',
                'slope_sn_col': self.slope_sn_col or 'none',
            }
            if era5 is not None:
                payload['era5'] = era5
            self._save_result(out_file, payload)
            _LOG.info('Saved %s', out_file.name)

            return {
                'status': 'success',
                'file': out_file,
                'n_stations': len(day_stations),
                'temp_range': (float(np.nanmin(final)), float(np.nanmax(final))),
                'uncertainty_range': (float(np.nanmin(uncertainty)), float(np.nanmax(uncertainty))),
            }

        except Exception as exc:
            _LOG.error('Failed for %s: %s', date_str, exc, exc_info=True)
            return {'status': 'error', 'error': str(exc)}
