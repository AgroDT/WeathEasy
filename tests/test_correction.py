"""Offline correction on coordinate-aware Zarr groups."""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from fsspec.implementations.memory import MemoryFileSystem
from zarr.abc.store import Store

from weatheasy import Coords, get_cfs2_data, get_era5_data
from weatheasy.build_copdem import _slopes
from weatheasy.correction import Source, apply, features, training
from weatheasy.correction.artifacts import Artifacts, get_artifacts
from weatheasy.correction.climatology import build_climatology
from weatheasy.correction.precip import WET_FEAT, PrecipCorrector
from weatheasy.correction.sources import CFS2, ERA5
from weatheasy.correction.temperature import FEATURE_COLS, TemperatureCorrector
from weatheasy.error import CoordsError
from weatheasy.storage import get_store, open_dataset, prepare_dataset, write_batch
from weatheasy.util import get_storage


class Constant:
    def __init__(self, value: Any) -> None:
        self.value = value

    def predict(self, x: np.ndarray):
        return np.full(len(x), self.value, np.float32)


class AlwaysWet:
    def predict_proba(self, x: np.ndarray):
        return np.tile([0.1, 0.9], (len(x), 1))


class RecordTemperature:
    def predict(self, x: np.ndarray):
        self.captured = x.copy()
        return np.full(len(x), 1.0, np.float32)


class RecordWet(AlwaysWet):
    def predict_proba(self, x: np.ndarray):
        self.captured = x.copy()
        return super().predict_proba(x)


def dem_store(path: Path, lats: np.ndarray, lons: np.ndarray) -> Store:
    shape = (len(lats), len(lons))
    store = get_store(path)
    xr.Dataset(
        {
            'elevation': (('latitude', 'longitude'), np.full(shape, 150, np.float32)),
            'slope_we': (('latitude', 'longitude'), np.zeros(shape, np.float32)),
            'slope_sn': (('latitude', 'longitude'), np.zeros(shape, np.float32)),
        },
        coords={'latitude': lats, 'longitude': lons},
    ).to_zarr(store, zarr_format=3, consolidated=False)
    return store


def raw_store(
    tmp_path: Path,
    source: Source = ERA5,
    days: int = 2,
    day_offsets: tuple[int, ...] | None = None,
    lon_start: float = 37,
):
    root = get_storage(str(tmp_path / 'data'))
    if source == ERA5:
        lats, lons = 55 + np.arange(11) * 0.1, lon_start + np.arange(11) * 0.1
    else:
        lats, lons = np.arange(40, 84), np.arange(18, 182)
    start = date(2020, 1, 1)
    offsets = tuple(range(days)) if day_offsets is None else day_offsets
    shape = (len(offsets), len(lats), len(lons))
    tmp = np.full(shape, 10, np.float32)
    # Raw daily samples are rates; keep their scale independent of Source.precip_to_mm.
    prate = np.full(shape, 10 / (86400 if source == ERA5 else 21600), np.float32)
    prate[0, 1, 1] = np.nan
    ds = xr.Dataset(
        {
            'TMP': (('time', 'latitude', 'longitude'), tmp, {'units': 'degC'}),
            'PRATE': (
                ('time', 'latitude', 'longitude'),
                prate,
                {'units': 'kg m-2 s-1'},
            ),
        },
        coords={
            'time': [np.datetime64(start) + np.timedelta64(n, 'D') for n in offsets],
            'latitude': lats,
            'longitude': lons,
        },
    )
    write_batch(root.store, source.path(root), prepare_dataset(ds, crs='EPSG:4326'))
    if source == CFS2:
        root.require_group('cfs2').attrs['updated'] = (
            start + timedelta(days=max(offsets) + 4)
        ).isoformat()
    return root, start


def heterogeneous_store(tmp_path: Path, source: Source):
    root = get_storage(str(tmp_path / 'data'))
    if source == ERA5:
        lats, lons = [55.0, 56.0], [37.0, 38.0]
        station_lat, station_lon = 55.0, 37.0
    else:
        lats, lons = [40.0, 41.0, 42.0], [18.0, 19.0, 20.0]
        station_lat, station_lon = 41.0, 19.0
    tmp = np.full((3, len(lats), len(lons)), 20.0, np.float32)
    precip = np.full_like(tmp, 8.0)
    li, lo = (0, 0) if source == ERA5 else (1, 1)
    tmp[:, li, lo] = [7.0, 13.0, np.nan]
    precip[:, li, lo] = [0.0, 8.0, np.nan]
    ds = xr.Dataset(
        {
            'TMP': (('time', 'latitude', 'longitude'), tmp, {'units': 'degC'}),
            'PRATE': (
                ('time', 'latitude', 'longitude'),
                precip / (86400 if source == ERA5 else 21600),
                {'units': 'kg m-2 s-1'},
            ),
        },
        coords={
            'time': np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[D]'),
            'latitude': lats,
            'longitude': lons,
        },
    )
    write_batch(root.store, source.path(root), prepare_dataset(ds, crs='EPSG:4326'))
    station = pd.DataFrame(
        {
            'station_id': [1],
            'lat': [station_lat],
            'lon': [station_lon],
            'date': [pd.Timestamp('2020-01-02')],
            'temperature': [12.0],
            'precipitation': [7.0],
            'elevation': [150.0],
            'slope_we': [0.0],
            'slope_sn': [0.0],
        }
    )
    return root, station, (0, 0)


def stub_models(monkeypatch: pytest.MonkeyPatch):
    class FakeTemperature:
        artifact_id = 'test-artifact'
        climatology = None

        def __init__(self, source: Source, _artifacts_dir, _dem_store) -> None:  # noqa: ANN001
            self.source = source

        def load_artifacts(self):
            return self

        def correct(self, root: zarr.Group, day: date):
            values, _, _ = self.source.read_day(root, 'TMP', day)
            return {'corrected': values - 2}

    class FakePrecipitation(FakeTemperature):
        def correct(self, root: zarr.Group, day: date):
            values, _, _ = self.source.read_day(root, 'PRATE', day)
            return {'corrected': values * self.source.precip_to_mm / 2}

    monkeypatch.setattr(apply, 'TemperatureCorrector', FakeTemperature)
    monkeypatch.setattr(apply, 'PrecipCorrector', FakePrecipitation)


def test_correction_append_and_rebuild_resets_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root, first = raw_store(tmp_path)
    stub_models(monkeypatch)
    args = (ERA5, root, get_artifacts(tmp_path / 'artifacts'), get_store(tmp_path / 'dem.zarr'))
    assert apply.correct_store(*args, end=first) == 1
    assert apply.correct_store(*args) == 1
    assert apply.correct_store(*args) == 0
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 2
        assert ds.PRATE_corr.attrs['units'] == 'mm/day'
    result = get_era5_data(
        store=root.store,
        begin=first,
        end=first,
        coords=Coords(55.5, 37.5),
        variables=['TMP', 'TMP_corr', 'PRATE_corr'],
    )
    np.testing.assert_allclose(result[:, 0], [10, 8, 5])
    # Losing either field forces both fields to be rebuilt from the first day.
    del root[ERA5.correction_path]['TMP_corr']
    assert apply.correct_store(*args) == 2
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 2
        np.testing.assert_allclose(ds.TMP_corr[:, 5, 5], [8, 8])


def test_cfs_cropped_correction_has_own_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, first = raw_store(tmp_path, CFS2)
    stub_models(monkeypatch)
    assert (
        apply.correct_store(CFS2, root, get_artifacts(tmp_path), get_store(tmp_path / 'dem.zarr'))
        == 2
    )
    inside = get_cfs2_data(
        store=root.store,
        begin=first,
        end=first,
        coords=Coords(55, 37),
        variables=['TMP', 'TMP_corr', 'PRATE_corr'],
    )
    np.testing.assert_allclose(inside[:, 0], [10, 8, 5])
    outside = get_cfs2_data(
        store=root.store,
        begin=first,
        end=first,
        coords=Coords(40, 18),
        variables=['TMP'],
    )
    np.testing.assert_allclose(outside, [[10]])
    with pytest.raises(CoordsError):
        get_cfs2_data(
            store=root.store,
            begin=first,
            end=first,
            coords=Coords(40, 18),
            variables=['TMP_corr'],
        )


def test_cfs_correction_resumes_across_missing_calendar_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, first = raw_store(tmp_path, CFS2, day_offsets=(0, 2))
    stub_models(monkeypatch)
    args = (CFS2, root, get_artifacts(tmp_path), get_store(tmp_path / 'dem.zarr'))
    assert apply.correct_store(*args, end=first) == 1
    assert apply.correct_store(*args) == 1
    assert apply.correct_store(*args) == 0
    with open_dataset(root.store, CFS2.corrected_path(root)) as ds:
        np.testing.assert_array_equal(
            ds.time.to_numpy().astype('datetime64[D]'),
            [np.datetime64(first), np.datetime64(first + timedelta(days=2))],
        )
    result = get_cfs2_data(
        store=root.store,
        begin=first,
        end=first + timedelta(days=2),
        coords=Coords(55, 37),
        variables=['TMP', 'TMP_corr', 'PRATE_corr'],
    )
    np.testing.assert_allclose(result[:, [0, 2]], [[10, 10], [8, 8], [5, 5]])
    assert np.isnan(result[:, 1]).all()


def test_correction_rejects_skipping_prepared_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, first = raw_store(tmp_path, days=3)
    stub_models(monkeypatch)
    args = (ERA5, root, get_artifacts(tmp_path), get_store(tmp_path / 'dem.zarr'))
    assert apply.correct_store(*args, end=first) == 1
    with pytest.raises(ValueError, match='cannot skip prepared day 2020-01-02'):
        apply.correct_store(*args, begin=first + timedelta(days=2))


def test_missing_precipitation_stays_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, first = raw_store(tmp_path)
    lats, lons = ERA5.grid(root)
    corrector = PrecipCorrector(
        ERA5, get_artifacts(tmp_path), dem_store(tmp_path / 'dem.zarr', lats, lons)
    )
    corrector.clf = corrector.fallback_clf = AlwaysWet()
    corrector.reg = corrector.fallback_reg = Constant(np.log1p(8))
    corrector.train_coords = np.array([[55.5, 37.5]], np.float32)
    corrector.grid_lats, corrector.grid_lons = lats, lons
    corrector.climatology = build_climatology(
        ERA5, root, ERA5.precip_var, ERA5.precip_to_mm, wet_threshold=1.0
    )
    corrector.seasonal = {
        'log_ratio': {m: np.zeros((len(lats), len(lons)), np.float32) for m in range(1, 13)},
        'wet_bias': {m: np.full((len(lats), len(lons)), 0.5, np.float32) for m in range(1, 13)},
    }
    calibration_targets: list[float] = []

    def record_target(_features, target: float):  # noqa: ANN001
        calibration_targets.append(target)
        return 0.5

    monkeypatch.setattr(corrector, '_calibrate', record_target)
    result = corrector.correct(root, first)
    assert np.isnan(result['corrected'][1, 1])
    assert result['corrected'][0, 0] == pytest.approx(8)
    # Every known cell is wet; the missing cell must not dilute the wet fraction.
    assert calibration_targets == pytest.approx([0.65])
    assert result['original'][0, 0] == pytest.approx(10)


def test_temperature_corrector_applies_and_suppresses_small_bias(tmp_path: Path):
    root, first = raw_store(tmp_path)
    lats, lons = ERA5.grid(root)
    corrector = TemperatureCorrector(
        ERA5, get_artifacts(tmp_path), dem_store(tmp_path / 'dem.zarr', lats, lons)
    )
    corrector.model = Constant(1.5)
    corrector.grid_lats, corrector.grid_lons = lats, lons
    corrector.climatology = build_climatology(ERA5, root, ERA5.temp_var, 1.0)
    corrector.seasonal_bias = {m: np.ones((len(lats), len(lons)), np.float32) for m in range(1, 13)}
    corrected = corrector.correct(root, first)
    np.testing.assert_allclose(corrected['original'] - corrected['corrected'], 1.5)
    corrector.seasonal_bias[1][:] = 0.1
    suppressed = corrector.correct(root, first)
    np.testing.assert_allclose(suppressed['original'], suppressed['corrected'])


@pytest.mark.parametrize('source', [ERA5, CFS2])
def test_training_and_apply_use_same_frozen_cell_month_features(tmp_path: Path, source: Source):
    root, station, target = heterogeneous_store(tmp_path, source)
    day = date(2020, 1, 2)
    temp_climate = build_climatology(source, root, 'TMP', 1.0)
    precip_climate = build_climatology(
        source, root, 'PRATE', source.precip_to_mm, wet_threshold=1.0
    )
    li, lo = target
    assert temp_climate.mean[0, li, lo] == pytest.approx(10.0)
    assert precip_climate.mean[0, li, lo] == pytest.approx(4.0)
    assert precip_climate.require_wet_fraction()[0, li, lo] == pytest.approx(0.5)
    assert np.isnan(temp_climate.mean[1]).all()  # No February observations.

    temp_rows = training._temp_features(
        training.sample_reanalysis(source, root, station, 'TMP', 1.0), temp_climate
    )
    precip_rows = training._precip_features(
        training.sample_reanalysis(source, root, station, 'PRATE', source.precip_to_mm),
        precip_climate,
    )
    assert temp_rows.era5_anomaly.iloc[0] == pytest.approx(3.0)
    assert precip_rows.era5_precip_anom.iloc[0] == pytest.approx(4.0)
    assert precip_rows.era5_wet_fraction.iloc[0] == pytest.approx(0.5)

    temp_model = RecordTemperature()
    lats, lons = source.grid(root)
    terrain = dem_store(tmp_path / 'dem.zarr', lats, lons)
    temp = TemperatureCorrector(source, get_artifacts(tmp_path), terrain)
    temp.model = temp_model
    temp.climatology = temp_climate
    temp.grid_lats, temp.grid_lons = source.grid(root)
    temp.seasonal_bias = {m: np.ones_like(temp_climate.mean[0]) for m in range(1, 13)}
    temp.correct(root, day)
    index = FEATURE_COLS.index('era5_anomaly')
    assert temp_model.captured[0, index] == pytest.approx(temp_rows.era5_anomaly.iloc[0])

    wet_model = RecordWet()
    precip = PrecipCorrector(source, get_artifacts(tmp_path), terrain)
    precip.clf = precip.fallback_clf = wet_model
    precip.reg = precip.fallback_reg = Constant(np.log1p(8))
    precip.climatology = precip_climate
    precip.grid_lats, precip.grid_lons = source.grid(root)
    precip.train_coords = station[['lat', 'lon']].to_numpy()
    precip.seasonal = {
        'log_ratio': {m: np.zeros_like(precip_climate.mean[0]) for m in range(1, 13)},
        'wet_bias': {m: np.full_like(precip_climate.mean[0], 0.5) for m in range(1, 13)},
    }
    precip.correct(root, day)
    assert wet_model.captured[0, WET_FEAT.index('era5_precip_anom')] == pytest.approx(
        precip_rows.era5_precip_anom.iloc[0]
    )
    assert wet_model.captured[0, WET_FEAT.index('era5_wet_fraction')] == pytest.approx(
        precip_rows.era5_wet_fraction.iloc[0]
    )

    # New raw days do not silently change features until a new training run.
    extra = xr.Dataset(
        {
            'TMP': (
                ('time', 'latitude', 'longitude'),
                np.full((1, *temp_climate.mean.shape[1:]), 100.0, np.float32),
                {'units': 'degC'},
            ),
            'PRATE': (
                ('time', 'latitude', 'longitude'),
                np.full(
                    (1, *temp_climate.mean.shape[1:]),
                    100.0 / source.precip_to_mm,
                    np.float32,
                ),
                {'units': 'kg m-2 s-1'},
            ),
        },
        coords={
            'time': [np.datetime64('2020-01-04')],
            'latitude': temp_climate.lats,
            'longitude': temp_climate.lons,
        },
    )
    # CFS2 raw contains the full grid; its cropped correction grid cannot append raw data.
    if source == ERA5:
        write_batch(root.store, source.path(root), prepare_dataset(extra, crs='EPSG:4326'))
        temp.correct(root, day)
        assert temp_model.captured[0, index] == pytest.approx(3.0)
        february = extra.assign_coords(time=[np.datetime64('2020-02-01')])
        write_batch(root.store, source.path(root), prepare_dataset(february, crs='EPSG:4326'))
        new_day = date(2020, 2, 1)
        temp_result = temp.correct(root, new_day)
        precip_result = precip.correct(root, new_day)
        assert np.isnan(temp_result['corrected']).all()
        assert np.isnan(precip_result['corrected']).all()
        assert np.isfinite(temp_result['original']).all()
        assert np.isfinite(precip_result['original']).all()


def test_training_discards_missing_observation_and_reanalysis(tmp_path: Path):
    csv = tmp_path / 'stations.csv'
    pd.DataFrame(
        {
            'station_id': [1, 2, 3],
            'lat': [55, 55, 55.1],
            'lon': [37, 37, 37.1],
            'date': ['2020-01-01'] * 3,
            'precipitation': [np.nan, 0, 3],
        }
    ).to_csv(csv, index=False)
    loaded = training.load_stations(csv, ['precipitation'])
    assert len(loaded) == 2
    root, _ = raw_store(tmp_path)
    sampled = training.sample_reanalysis(ERA5, root, loaded, 'PRATE', 86400)
    assert len(sampled) == 1  # missing raw cell is not used as a dry observation
    assert sampled['precipitation'].iloc[0] == 0
    assert np.isfinite(sampled['reanalysis']).all()


def test_training_reports_and_skips_stations_outside_cell_bounds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level('INFO', logger='weatheasy.correction.training')
    root, first = raw_store(tmp_path, lon_start=179)
    stations = pd.DataFrame(
        {
            'station_id': [1, 1, 2, 2, 3, 3],
            'lat': [55.5, 55.5, 54.94, 54.94, 55.5, 55.5],
            'lon': [179.5, 179.5, 179.5, 179.5, -180.0, -180.0],
            'date': [pd.Timestamp(first), pd.Timestamp(first + timedelta(days=1))] * 3,
        }
    )
    sampled = training.sample_reanalysis(ERA5, root, stations, 'TMP', 1)
    assert set(sampled.station_id) == {1, 3}
    assert len(sampled) == 4
    assert 'station 2 (lat=54.94, lon=179.5) outside prepared grid bounds' in caplog.text
    assert caplog.text.count('station 2 (') == 1
    assert 'skipping 2 station-days' in caplog.text
    assert '1 outside; skipped 2 station-days' in caplog.text
    with pytest.raises(RuntimeError, match='No station-days inside'):
        training.sample_reanalysis(ERA5, root, stations[stations.station_id == 2], 'TMP', 1)


def test_station_terrain_enrichment_and_source_units(tmp_path: Path):
    root, first = raw_store(tmp_path)
    station = pd.DataFrame(
        {
            'station_id': [1],
            'lat': [55.5],
            'lon': [37.5],
            'date': [pd.Timestamp(first)],
            'temperature': [8.0],
        }
    )
    lats, lons = ERA5.grid(root)
    dem: features.Dem = {
        'lats': lats,
        'lons': lons,
        'elevation': np.full((len(lats), len(lons)), 150, np.float32),
        'slope_we': np.zeros((len(lats), len(lons)), np.float32),
        'slope_sn': np.zeros((len(lats), len(lons)), np.float32),
    }
    enriched = training.enrich_stations(station, dem)
    assert enriched['elevation'].iloc[0] == 150
    sampled = training.sample_reanalysis(ERA5, root, enriched, 'TMP', 1.0)
    assert sampled['reanalysis'].iloc[0] == 10
    raw, source_lats, source_lons = ERA5.read_day(root, 'PRATE', first)
    assert source_lats[0] == pytest.approx(55)
    assert source_lons[-1] == pytest.approx(38)
    assert raw[5, 5] * ERA5.precip_to_mm == pytest.approx(10)


def test_dem_missing_cell_cannot_become_zero_terrain():
    lats = np.array([55.0, 55.1], np.float32)
    lons = np.array([37.0, 37.1], np.float32)
    dem: features.Dem = {
        'lats': lats,
        'lons': lons,
        'elevation': np.array([[150, np.nan], [150, 150]], np.float32),
        'slope_we': np.zeros((2, 2), np.float32),
        'slope_sn': np.zeros((2, 2), np.float32),
    }
    with pytest.raises(ValueError, match='DEM field elevation contains missing values'):
        features.resample_dem(dem, lats, lons)


def test_longitude_scale_changes_hybrid_distance_and_idw_weights():
    target = np.array([[60.0, 0.0]])
    refs = np.array([[60.0, 2.0], [61.0, 0.0], [62.0, 0.0]])
    distance = features.point_distances_km(target, refs)[0]
    np.testing.assert_allclose(distance, [111, 111, 222], rtol=1e-12)
    assert distance[0] < 200  # unscaled longitude would be 222 km, selecting fallback
    frame = pd.DataFrame(
        {
            'station_id': [1, 2, 3],
            'month': [1, 1, 1],
            'lat': refs[:, 0],
            'lon': refs[:, 1],
            'bias': [0, 10, 0],
        }
    )
    seasonal = training.idw_seasonal(frame, 'bias', np.array([60.0]), np.array([0.0]))
    assert seasonal[1][0, 0] == pytest.approx(10 / 2.25)  # 1:1:1/4 inverse-square weights


def test_dem_slope_uses_longitude_scale_at_60_degrees():
    elevation = np.array([[0, 100], [0, 100]], dtype=np.float32)
    west_east, south_north = _slopes(elevation, np.array([60.0, 60.1]), 0.1)
    assert west_east[0, 0] == pytest.approx(100 / (0.1 * 111320 * 0.5))
    np.testing.assert_allclose(south_north, 0)


def test_separate_training_and_apply_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from weatheasy.correction.__main__ import main

    root, first = raw_store(tmp_path)
    dem_path = tmp_path / 'dem.zarr'
    dem_store(dem_path, *ERA5.grid(root))
    stations = tmp_path / 'stations.csv'
    rows = [
        (
            station,
            55.1 + station * 0.1,
            37.1 + station * 0.1,
            (first + timedelta(days=day)).isoformat(),
            8.0,
            3.0 if station % 2 else 0.0,
        )
        for station in range(6)
        for day in range(2)
    ]
    pd.DataFrame(
        rows,
        columns=['station_id', 'lat', 'lon', 'date', 'temperature', 'precipitation'],
    ).to_csv(stations, index=False)
    monkeypatch.setitem(training._XGB_PARAMS, 'n_estimators', 8)
    common = [
        'era5',
        '--store',
        str(tmp_path / 'data'),
        '--dem',
        str(dem_path),
        '--artifacts',
        str(tmp_path / 'artifacts'),
    ]
    main(['train', *common, '--stations', str(stations)])
    main(['apply', *common, '--end', first.isoformat()])
    old_id = root[ERA5.correction_path].attrs['correction_artifact_id']
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 1
        assert np.isfinite(ds.TMP_corr).all()
        assert float(ds.TMP_corr.mean()) < 10
        assert np.isnan(ds.PRATE_corr[0, 1, 1])
    main(['train', *common, '--stations', str(stations), '--only', 'temp'])
    main(['apply', *common])
    assert root[ERA5.correction_path].attrs['correction_artifact_id'] != old_id
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 2  # Rebuilt from the first day with the new model pair.


def test_old_artifacts_are_rejected_before_correction_is_deleted(tmp_path: Path):
    root, first = raw_store(tmp_path)
    artifacts = tmp_path / 'artifacts'
    old_dir = artifacts / ERA5.name / 'temp'
    old_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match='old or incomplete correction artifacts; retrain'):
        apply.correct_store(
            ERA5, root, get_artifacts(artifacts), get_store(tmp_path / 'missing.zarr'), end=first
        )
    assert ERA5.correction_path not in root


def test_interrupted_retraining_artifacts_are_rejected(tmp_path: Path):
    from weatheasy.correction.climatology import TEMP_FILES, verify_manifest, write_manifest

    directory = tmp_path / 'artifacts'
    directory.mkdir()
    for name in TEMP_FILES:
        (directory / name).write_bytes(b'original')
    write_manifest(get_artifacts(directory), TEMP_FILES)
    verify_manifest(get_artifacts(directory), TEMP_FILES)
    (directory / 'model.joblib').write_bytes(b'interrupted replacement')
    with pytest.raises(ValueError, match='incomplete or modified correction artifacts'):
        verify_manifest(get_artifacts(directory), TEMP_FILES)


@pytest.fixture(params=['local', 'memory'])
def artifact_directory(request: pytest.FixtureRequest, tmp_path: Path) -> Artifacts:
    if request.param == 'local':
        return get_artifacts(tmp_path / 'models')
    return Artifacts(MemoryFileSystem(), f'/models/{tmp_path.name}')


def test_filesystem_model_training_application_and_resume(
    artifact_directory: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, first = raw_store(tmp_path)
    terrain = dem_store(tmp_path / 'dem.zarr', *ERA5.grid(root))
    stations = tmp_path / 'stations.csv'
    pd.DataFrame(
        [
            (
                station,
                55.1 + station * 0.1,
                37.1 + station * 0.1,
                (first + timedelta(days=day)).isoformat(),
                8.0,
                3.0 if station % 2 else 0.0,
            )
            for station in range(6)
            for day in range(2)
        ],
        columns=['station_id', 'lat', 'lon', 'date', 'temperature', 'precipitation'],
    ).to_csv(stations, index=False)
    monkeypatch.setitem(training._XGB_PARAMS, 'n_estimators', 8)
    training.build_artifacts(ERA5, root, stations, terrain, artifact_directory)
    # A fresh reader has no handles or in-memory model state from training.
    fresh = Artifacts(artifact_directory.fs, artifact_directory.root)
    assert apply.correct_store(ERA5, root, fresh, terrain, end=first) == 1
    assert apply.correct_store(ERA5, root, fresh, terrain) == 1
    assert apply.correct_store(ERA5, root, fresh, terrain) == 0
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        corrected = ds.TMP_corr.to_numpy()
        assert ds.sizes['time'] == 2
        assert np.isfinite(ds.TMP_corr[0, 0, 0])
        assert float(ds.TMP_corr[0, 0, 0]) < 10
        assert np.isnan(ds.PRATE_corr[0, 1, 1])
    with fresh.child(ERA5.name, 'temp').open('model.joblib', 'wb') as stream:
        stream.write(b'interrupted retraining')
    with pytest.raises(ValueError, match='incomplete or modified correction artifacts'):
        apply.correct_store(ERA5, root, fresh, terrain)
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        np.testing.assert_array_equal(ds.TMP_corr.to_numpy(), corrected)


@pytest.mark.parametrize('damage', ['missing-manifest', 'missing-file', 'hash', 'schema'])
def test_manifest_validates_filesystem_bundle(artifact_directory: Artifacts, damage: str):
    from weatheasy.correction.climatology import verify_manifest, write_manifest

    directory = artifact_directory
    directory.fs.makedirs(directory.root, exist_ok=True)
    for name, contents in [('a.joblib', b'first'), ('b.npz', b'second')]:
        with directory.open(name, 'wb') as stream:
            stream.write(contents)
    files = ('a.joblib', 'b.npz')
    write_manifest(directory, files)
    with directory.open('manifest.json') as stream:
        manifest = json.load(stream)
    assert manifest == {
        'schema': 'cell_month_v1',
        'files': {
            'a.joblib': hashlib.sha256(b'first').hexdigest(),
            'b.npz': hashlib.sha256(b'second').hexdigest(),
        },
    }
    verify_manifest(directory, files)
    if damage == 'missing-manifest':
        directory.fs.rm(directory.child('manifest.json').root)
        message = 'old or incomplete correction artifacts'
    elif damage == 'missing-file':
        directory.fs.rm(directory.child('b.npz').root)
        message = 'incomplete or modified correction artifacts'
    elif damage == 'hash':
        with directory.open('a.joblib', 'wb') as stream:
            stream.write(b'replaced')
        message = 'incomplete or modified correction artifacts'
    else:
        manifest['schema'] = 'incompatible'
        with directory.open('manifest.json', 'wb') as stream:
            stream.write(json.dumps(manifest).encode())
        message = 'incompatible correction artifacts'
    with pytest.raises(ValueError, match=message):
        verify_manifest(directory, files)


def test_failed_manifest_publication_keeps_previous_manifest(
    artifact_directory: Artifacts, monkeypatch: pytest.MonkeyPatch
):
    from weatheasy.correction import climatology

    directory = artifact_directory
    directory.fs.makedirs(directory.root, exist_ok=True)
    with directory.open('model.joblib', 'wb') as stream:
        stream.write(b'original')
    climatology.write_manifest(directory, ('model.joblib',))
    with directory.open('manifest.json') as stream:
        previous = stream.read()
    with directory.open('model.joblib', 'wb') as stream:
        stream.write(b'retrained')

    def failed_serialization(*_args, **_kwargs):
        msg = 'interrupted publication'
        raise RuntimeError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(climatology.json, 'dumps', failed_serialization)
        with pytest.raises(RuntimeError, match='interrupted publication'):
            climatology.write_manifest(directory, ('model.joblib',))
    with directory.open('manifest.json') as stream:
        assert stream.read() == previous
    with pytest.raises(ValueError, match='incomplete or modified correction artifacts'):
        climatology.verify_manifest(directory, ('model.joblib',))
