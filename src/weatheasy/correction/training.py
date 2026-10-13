"""Offline training of the bias-correction artifacts (``build_artifacts``).

Reads terrain-enriched station observations and historical reanalysis from Zarr,
and writes per-source model artifacts consumed by the inference correctors.
This is the heavy offline step, not part of the realtime path.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from . import features, quality
from .climatology import (
    PRECIP_FILES,
    TEMP_FILES,
    Climatology,
    build_climatology,
    write_manifest,
)
from .precip import (
    _SEASONAL_FEAT,
    HYBRID_RADIUS_KM,
    WET_BIAS_THRESHOLD,
    WET_FEAT,
    WET_THRESHOLD,
    _predict,
)
from .sources import FORECASTS
from .temperature import CORRECTION_THRESHOLD, FEATURE_COLS
from weatheasy.storage import StorageError, select_point


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import zarr
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)

_TERRAIN_COLS = ('elevation', 'slope_we', 'slope_sn')
_IDW_BATCH = 100_000
_IDW_MIN_STATIONS = 3
CV_FOLDS = 5
_XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'device': 'cpu',
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
}


# ── Data preparation ────────────────────────────────────────────────────────


def prepare_stations(observations: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    """Normalize station observations without changing the caller's table."""
    stations = observations.copy()
    if 'date' in stations.columns:
        stations['date'] = pd.to_datetime(stations['date'])
    else:
        stations['date'] = pd.to_datetime(stations[['year', 'month', 'day']])
    keep = ['station_id', 'lat', 'lon', 'date', *value_cols, *_TERRAIN_COLS]
    missing = set(keep) - set(stations.columns)
    if missing:
        msg = f'Station observations are missing required columns: {sorted(missing)}'
        raise ValueError(msg)
    stations = stations[keep]
    stations = stations.dropna(subset=keep)
    for col in ('lat', 'lon', *value_cols):
        if col in stations.columns:
            stations[col] = stations[col].astype(np.float32)
    _LOG.info(
        'Loaded %d station-days, %d stations',
        len(stations),
        stations['station_id'].nunique(),
    )
    return stations


def sample_reanalysis(
    source: Source,
    root: zarr.Group,
    stations: pd.DataFrame,
    variable: str,
    scale: float,
) -> pd.DataFrame:
    """Sample the daily reanalysis value at each station-day from the Zarr store.

    Reads one day-slice per date and nearest-neighbour samples stations inside
    the grid's half-cell bounds. Dates absent from the store are skipped.
    """
    lats, lons = source.grid(root)
    coordinates = xr.Dataset(
        coords={'latitude': lats.astype(np.float64), 'longitude': lons.astype(np.float64)},
    )
    station_keys = list(stations[['station_id', 'lat', 'lon']].itertuples(index=False, name=None))
    counts = Counter(station_keys)
    positions: dict[tuple, tuple[int, int]] = {}
    skipped = 0
    south = float(lats[0] - (lats[1] - lats[0]) / 2)
    north = float(lats[-1] + (lats[-1] - lats[-2]) / 2)
    west = float(lons[0] - (lons[1] - lons[0]) / 2)
    east = float(lons[-1] + (lons[-1] - lons[-2]) / 2)
    for key, count in counts.items():
        station_id, latitude, longitude = key
        try:
            point = select_point(
                coordinates,
                latitude=float(latitude),
                longitude=float(longitude),
            )
        except StorageError:
            skipped += count
            _LOG.warning(
                '%s: station %s (lat=%s, lon=%s) outside prepared grid bounds '
                'lat=[%.5f, %.5f], lon=[%.5f, %.5f]; skipping %d station-days',
                source.name,
                station_id,
                latitude,
                longitude,
                south,
                north,
                west,
                east,
                count,
            )
            continue
        positions[key] = (
            int(np.searchsorted(lats, point.latitude.item())),
            int(np.searchsorted(lons, point.longitude.item())),
        )
    _LOG.info(
        '%s: station coverage: %d inside, %d outside; skipped %d station-days',
        source.name,
        len(positions),
        len(counts) - len(positions),
        skipped,
    )
    if not positions:
        msg = f'No station-days inside the prepared {source.name} grid'
        raise RuntimeError(msg)
    stations = stations.loc[[key in positions for key in station_keys]]
    chunks = []
    for day, group in stations.groupby('date'):
        if not isinstance(day, datetime):
            msg = 'Station dates must be datetimes'
            raise TypeError(msg)
        try:
            grid, _, _ = source.read_day(root, variable, day.date())
        except IndexError:
            continue
        indices = [
            positions[key]
            for key in group[['station_id', 'lat', 'lon']].itertuples(index=False, name=None)
        ]
        li, lo = np.asarray(indices).T
        out = group.copy()
        out['reanalysis'] = (grid[li, lo] * scale).astype(np.float32)
        out['_lat_index'] = li
        out['_lon_index'] = lo
        out = out[np.isfinite(out['reanalysis'])]
        chunks.append(out)
    if not chunks or not any(len(chunk) for chunk in chunks):
        msg = 'No station-days overlapped the reanalysis store'
        raise RuntimeError(msg)
    result = pd.concat(chunks, ignore_index=True)
    _LOG.info('Sampled reanalysis for %d station-days', len(result))
    return result


# ── Temperature training ────────────────────────────────────────────────────


def build_temp_artifacts(
    source: Source,
    root: zarr.Group,
    observations: pd.DataFrame,
    artifacts: FileStore,
    *,
    cv_folds: int = CV_FOLDS,
    dem_store: Store | None = None,
) -> None:
    """Train the temperature corrector for ``source`` and write its artifacts."""
    stations = prepare_stations(observations, ['temperature'])
    sampled = sample_reanalysis(source, root, stations, source.temp_var, 1.0)
    climate = build_climatology(source, root, source.temp_var, 1.0)
    dataset = _temp_features(sampled, climate)
    model = _train_bias_regressor(dataset)
    lats, lons = source.grid(root)
    seasonal = idw_seasonal(dataset, 'era5_seasonal_bias', lats, lons)

    out = artifacts.child(source.name, 'temp')
    out.fs.makedirs(out.root, exist_ok=True)
    climate.metadata['training_id'] = uuid.uuid4().hex
    climate.save(out)
    with out.open('model.joblib', 'wb') as stream:
        joblib.dump(model, stream)
    with out.open('seasonal_bias_grid.npz', 'wb') as stream:
        np.savez_compressed(
            stream, allow_pickle=False, **{f'month_{m}': seasonal[m] for m in range(1, 13)}
        )
    with out.open('lats.npy', 'wb') as stream:
        np.save(stream, lats)
    with out.open('lons.npy', 'wb') as stream:
        np.save(stream, lons)
    write_manifest(out, TEMP_FILES)
    _LOG.info('Wrote temperature artifacts for %s to %s', source.name, out.root)
    _report(
        source,
        root,
        artifacts,
        dem_store,
        'temp',
        cross_validate_temp(dataset, cv_folds),
        observed='temperature',
    )


def _temp_features(sampled: pd.DataFrame, climate: Climatology) -> pd.DataFrame:
    dataset = sampled.copy()
    dataset['bias'] = (dataset['reanalysis'] - dataset['temperature']).astype(np.float32)
    dataset['month'] = dataset['date'].dt.month.astype(np.int8)
    doy = dataset['date'].dt.dayofyear.to_numpy()
    dataset['doy_sin'] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    dataset['doy_cos'] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    dataset['era5_value'] = dataset['reanalysis']
    clim = climate.sample(
        dataset['month'].to_numpy(),
        dataset['_lat_index'].to_numpy(),
        dataset['_lon_index'].to_numpy(),
    )
    dataset['era5_anomaly'] = (dataset['reanalysis'].to_numpy() - clim).astype(np.float32)
    seasonal_bias = dataset.groupby(['station_id', 'month'])['bias'].transform('mean')
    dataset['era5_seasonal_bias'] = seasonal_bias.astype(np.float32)
    return dataset


def _train_bias_regressor(dataset: pd.DataFrame) -> Any:
    x = dataset[FEATURE_COLS].astype(np.float32).to_numpy()
    y = dataset['bias'].astype(np.float32).to_numpy()
    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(x, y, verbose=False)
    _LOG.info('Trained temperature regressor on %d rows', len(dataset))
    return model


def idw_seasonal(
    dataset: pd.DataFrame,
    column: str,
    lats: NDArray,
    lons: NDArray,
) -> dict[int, NDArray[np.float32]]:
    """Inverse-distance interpolate a per-station-month value onto the source grid."""
    station_month = _station_month(dataset, column)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    target = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    result: dict[int, NDArray[np.float32]] = {}
    for month in range(1, 13):
        month_data = station_month[station_month['month'] == month]
        if len(month_data) < _IDW_MIN_STATIONS:
            result[month] = np.zeros((len(lats), len(lons)), np.float32)
            continue
        result[month] = _idw(month_data, target).reshape(len(lats), len(lons))
    return result


def idw_points(dataset: pd.DataFrame, column: str, points: pd.DataFrame) -> NDArray[np.float32]:
    """Interpolate a per-station-month value of ``dataset`` to each row of ``points``.

    Matches :func:`idw_seasonal` sampled at the points, so held-out stations get
    the seasonal features inference would give them, not their own means.
    """
    station_month = _station_month(dataset, column)
    keys = points[['lat', 'lon', 'month']].drop_duplicates()
    values = np.zeros(len(keys), np.float32)
    months = keys['month'].to_numpy()
    for month in np.unique(months):
        month_data = station_month[station_month['month'] == month]
        if len(month_data) < _IDW_MIN_STATIONS:
            continue
        rows = months == month
        values[rows] = _idw(month_data, keys.loc[rows, ['lat', 'lon']].to_numpy(np.float64))
    keyed = keys.assign(_value=values)
    merged = points[['lat', 'lon', 'month']].merge(keyed, on=['lat', 'lon', 'month'], how='left')
    return merged['_value'].to_numpy(np.float32)


def _station_month(dataset: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        dataset.groupby(['station_id', 'month'])
        .agg(lat=('lat', 'first'), lon=('lon', 'first'), value=(column, 'mean'))
        .reset_index()
    )


def _idw(month_data: pd.DataFrame, target: NDArray) -> NDArray[np.float32]:
    src = month_data[['lat', 'lon']].to_numpy()
    vals = month_data['value'].to_numpy().astype(np.float32)
    interp = np.empty(len(target), np.float32)
    for s in range(0, len(target), _IDW_BATCH):
        e = min(s + _IDW_BATCH, len(target))
        dist = np.maximum(features.point_distances_km(target[s:e], src), 0.1)
        weight = 1.0 / dist**2
        weight /= weight.sum(axis=1, keepdims=True)
        interp[s:e] = (weight * vals[None, :]).sum(axis=1)
    return interp


# ── Precipitation training ──────────────────────────────────────────────────

_PRECIP_SEASONAL = ['era5_seasonal_log_ratio', 'precip_wet_bias']


def build_precip_artifacts(
    source: Source,
    root: zarr.Group,
    observations: pd.DataFrame,
    artifacts: FileStore,
    *,
    cv_folds: int = CV_FOLDS,
    dem_store: Store | None = None,
) -> None:
    """Train the two-stage precipitation corrector for ``source`` and write its artifacts."""
    stations = prepare_stations(observations, ['precipitation'])
    sampled = sample_reanalysis(source, root, stations, source.precip_var, source.precip_to_mm)
    climate = build_climatology(
        source, root, source.precip_var, source.precip_to_mm, wet_threshold=WET_THRESHOLD
    )
    dataset = _precip_features(sampled, climate)

    clf, reg = _train_two_stage(dataset)
    fallback = dataset.copy()
    fallback[_PRECIP_SEASONAL] = np.float32(0.0)
    fallback_clf, fallback_reg = _train_two_stage(fallback)
    threshold = _tune_threshold(clf, dataset)
    fallback_threshold = _tune_threshold(fallback_clf, fallback)

    lats, lons = source.grid(root)
    log_ratio = idw_seasonal(dataset, 'era5_seasonal_log_ratio', lats, lons)
    wet_bias = idw_seasonal(dataset, 'precip_wet_bias', lats, lons)
    train_coords = dataset[['lat', 'lon']].drop_duplicates().to_numpy().astype(np.float32)

    out = artifacts.child(source.name, 'precip')
    out.fs.makedirs(out.root, exist_ok=True)
    climate.metadata['training_id'] = uuid.uuid4().hex
    climate.save(out)
    with out.open('clf.joblib', 'wb') as stream:
        joblib.dump(clf, stream)
    with out.open('reg.joblib', 'wb') as stream:
        joblib.dump(reg, stream)
    with out.open('fallback_clf.joblib', 'wb') as stream:
        joblib.dump(fallback_clf, stream)
    with out.open('fallback_reg.joblib', 'wb') as stream:
        joblib.dump(fallback_reg, stream)
    with out.open('train_coords.npy', 'wb') as stream:
        np.save(stream, train_coords)
    with out.open('lats.npy', 'wb') as stream:
        np.save(stream, lats)
    with out.open('lons.npy', 'wb') as stream:
        np.save(stream, lons)
    seasonal: dict[str, Any] = {'threshold': threshold, 'fallback_threshold': fallback_threshold}
    for month in range(1, 13):
        seasonal[f'log_ratio_m{month}'] = log_ratio[month]
        seasonal[f'wet_bias_m{month}'] = wet_bias[month]
    with out.open('seasonal.npz', 'wb') as stream:
        np.savez_compressed(stream, **seasonal)
    write_manifest(out, PRECIP_FILES)
    _LOG.info('Wrote precip artifacts for %s to %s', source.name, out.root)
    _report(
        source,
        root,
        artifacts,
        dem_store,
        'precip',
        cross_validate_precip(dataset, cv_folds),
        observed='obs_precip',
    )


def _precip_features(sampled: pd.DataFrame, climate: Climatology) -> pd.DataFrame:
    dataset = sampled.copy()
    dataset['era5_precip'] = np.maximum(dataset['reanalysis'].to_numpy(), 0.0).astype(np.float32)
    dataset['obs_precip'] = np.maximum(dataset['precipitation'].to_numpy(), 0.0).astype(np.float32)
    dataset['era5_precip_log'] = np.log1p(dataset['era5_precip']).astype(np.float32)
    dataset['obs_precip_log'] = np.log1p(dataset['obs_precip']).astype(np.float32)
    dataset['obs_wet'] = (dataset['obs_precip'] >= WET_THRESHOLD).astype(np.int8)
    dataset['era5_wet'] = (dataset['era5_precip'] >= WET_THRESHOLD).astype(np.int8)
    dataset['month'] = dataset['date'].dt.month.astype(np.int8)
    doy = dataset['date'].dt.dayofyear.to_numpy()
    dataset['doy_sin'] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    dataset['doy_cos'] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    grp = ['station_id', 'month']
    month = dataset['month'].to_numpy()
    lat_index = dataset['_lat_index'].to_numpy()
    lon_index = dataset['_lon_index'].to_numpy()
    clim = climate.sample(month, lat_index, lon_index)
    dataset['era5_precip_anom'] = (dataset['era5_precip'].to_numpy() - clim).astype(np.float32)
    wet_fraction = climate.require_wet_fraction()
    dataset['era5_wet_fraction'] = wet_fraction[month - 1, lat_index, lon_index]
    wet_wet = (dataset['era5_wet'] == 1) & (dataset['obs_wet'] == 1)
    dataset['_log_ratio'] = np.where(
        wet_wet,
        dataset['obs_precip_log'] - dataset['era5_precip_log'],
        np.nan,
    )
    log_ratio = dataset.groupby(grp)['_log_ratio'].transform('mean')
    dataset['era5_seasonal_log_ratio'] = log_ratio.fillna(0).astype(np.float32)
    dataset['_wet_diff'] = (dataset['era5_wet'] - dataset['obs_wet']).astype(np.float32)
    wet_bias = dataset.groupby(grp)['_wet_diff'].transform('mean')
    dataset['precip_wet_bias'] = wet_bias.fillna(0).astype(np.float32)
    dataset['log_dist_coast'] = np.log1p(
        features.dist_to_coast(dataset['lat'].to_numpy(), dataset['lon'].to_numpy()),
    ).astype(np.float32)
    return dataset.drop(columns=['_log_ratio', '_wet_diff'])


def _train_two_stage(dataset: pd.DataFrame) -> tuple[Any, Any]:
    clf = xgb.XGBClassifier(**_XGB_PARAMS, eval_metric='logloss')
    clf.fit(
        dataset[WET_FEAT].astype(np.float32).to_numpy(),
        dataset['obs_wet'].to_numpy(),
        verbose=False,
    )
    wet = dataset[dataset['obs_wet'] == 1]
    reg = xgb.XGBRegressor(**_XGB_PARAMS)
    reg.fit(
        wet[WET_FEAT].astype(np.float32).to_numpy(),
        wet['obs_precip_log'].to_numpy(),
        verbose=False,
    )
    return clf, reg


def _tune_threshold(clf: Any, dataset: pd.DataFrame) -> float:
    probs = clf.predict_proba(dataset[WET_FEAT].astype(np.float32).to_numpy())[:, 1]
    target = float(dataset['obs_wet'].mean())
    best_t, best_score = 0.5, np.inf
    for t in np.linspace(0.1, 0.9, 81):
        score = abs(float((probs >= t).mean()) - target)
        if score < best_score:
            best_score, best_t = score, float(t)
    return best_t


# ── Orchestration ───────────────────────────────────────────────────────────


def build_artifacts(
    source: Source,
    root: zarr.Group,
    observations: pd.DataFrame,
    artifacts: FileStore,
    *,
    only: str | None = None,
    cv_folds: int = CV_FOLDS,
    dem_store: Store | None = None,
) -> None:
    """Train and write both correctors' artifacts for ``source`` (or just one).

    Each trained corrector also updates ``<source>/quality_report.json`` and the
    combined ``quality_report.md`` at the artifact root. ``cv_folds=0`` skips the
    cross-validated scores, which retrain the models once per fold.
    """
    if only in (None, 'temp'):
        build_temp_artifacts(
            source, root, observations, artifacts, cv_folds=cv_folds, dem_store=dem_store
        )
    if only in (None, 'precip'):
        build_precip_artifacts(
            source, root, observations, artifacts, cv_folds=cv_folds, dem_store=dem_store
        )


# ── Quality report ──────────────────────────────────────────────────────────


def _cv_splits(dataset: pd.DataFrame, folds: int) -> Iterator[tuple[NDArray, NDArray]] | None:
    """Station-grouped folds: a station is never in its own training data."""
    folds = min(folds, int(dataset['station_id'].nunique()))
    if folds < 2:
        return None
    return GroupKFold(n_splits=folds).split(dataset, groups=dataset['station_id'])


def cross_validate_temp(dataset: pd.DataFrame, folds: int) -> tuple[pd.DataFrame, int] | None:
    """Corrected temperature of each held-out station, as inference would produce it."""
    splits = _cv_splits(dataset, folds)
    if splits is None:
        return None
    corrected = np.full(len(dataset), np.nan, np.float32)
    count = 0
    for train_idx, test_idx in splits:
        count += 1
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx].copy()
        test['era5_seasonal_bias'] = idw_points(train, 'era5_seasonal_bias', test)
        model = _train_bias_regressor(train)
        bias = model.predict(test[FEATURE_COLS].astype(np.float32).to_numpy()).astype(np.float32)
        bias[np.abs(test['era5_seasonal_bias'].to_numpy()) < CORRECTION_THRESHOLD] = 0.0
        corrected[test_idx] = test['reanalysis'].to_numpy() - bias
    frame = dataset[['station_id', 'date', 'temperature']].assign(
        raw=dataset['reanalysis'].to_numpy(), corrected=corrected
    )
    return frame, count


def cross_validate_precip(dataset: pd.DataFrame, folds: int) -> tuple[pd.DataFrame, int] | None:
    """Corrected precipitation of each held-out station, as inference would produce it.

    Inference recalibrates the wet threshold per day from the whole grid; held-out
    stations use the thresholds tuned on the fold's training stations instead.
    """
    splits = _cv_splits(dataset, folds)
    if splits is None:
        return None
    corrected = np.full(len(dataset), np.nan, np.float32)
    count = 0
    for train_idx, test_idx in splits:
        count += 1
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx].copy()
        for column in _PRECIP_SEASONAL:
            test[column] = idw_points(train, column, test)
        clf, reg = _train_two_stage(train)
        fallback = train.copy()
        fallback[_PRECIP_SEASONAL] = np.float32(0.0)
        fallback_clf, fallback_reg = _train_two_stage(fallback)
        threshold = _tune_threshold(clf, train)
        fallback_threshold = _tune_threshold(fallback_clf, fallback)
        train_coords = train[['lat', 'lon']].drop_duplicates().to_numpy(np.float64)
        stations, inverse = np.unique(
            test[['lat', 'lon']].to_numpy(np.float64), axis=0, return_inverse=True
        )
        near_station = features.point_distances_km(stations, train_coords).min(axis=1)
        near = (near_station < HYBRID_RADIUS_KM)[inverse.ravel()]
        result = np.zeros(len(test), np.float32)
        if near.any():
            result[near] = _predict(test[near], clf, reg, threshold)
        if (~near).any():
            far = test[~near].copy()
            far[_SEASONAL_FEAT] = np.float32(0.0)
            result[~near] = _predict(far, fallback_clf, fallback_reg, fallback_threshold)
        suppress = np.abs(test['precip_wet_bias'].to_numpy()) < WET_BIAS_THRESHOLD
        result[suppress] = test['era5_precip'].to_numpy()[suppress]
        corrected[test_idx] = np.maximum(result, 0.0)
    frame = dataset[['station_id', 'date', 'obs_precip']].assign(
        raw=dataset['era5_precip'].to_numpy(), corrected=corrected
    )
    return frame, count


def _report(  # noqa: PLR0917
    source: Source,
    root: zarr.Group,
    artifacts: FileStore,
    dem_store: Store | None,
    variable: str,
    validated: tuple[pd.DataFrame, int] | None,
    *,
    observed: str,
) -> None:
    """Write the reanalysis scores and, for a source with a forecast, its shift check."""
    if validated is None:
        section: dict[str, Any] = {
            'skipped': 'cross-validation disabled or fewer than two stations',
        }
    else:
        frame, folds = validated
        section = quality.skill(frame, observed=observed, precipitation=variable == 'precip')
        section['folds'] = folds
        _LOG.info(
            '%s %s held-out RMSE: %.3f raw -> %.3f corrected',
            source.name,
            variable,
            section['raw']['rmse'],
            section['corrected']['rmse'],
        )
    quality.update_report(artifacts, source.name, variable, section)
    forecast = FORECASTS.get(source.name)
    if forecast is None or dem_store is None:
        return
    try:
        consistency = quality.forecast_consistency(forecast, root, artifacts, dem_store, variable)
    except (KeyError, OSError, ValueError, StorageError):
        # The models are already published; a broken forecast must not fail training.
        _LOG.exception('%s: forecast consistency check failed', forecast.name)
        return
    if consistency is not None:
        quality.update_report(artifacts, source.name, f'forecast_{variable}', consistency)
