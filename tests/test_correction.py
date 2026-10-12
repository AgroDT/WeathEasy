"""Offline correction on coordinate-aware Zarr groups."""

import hashlib
import json
from collections import Counter
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
from zarr.storage import MemoryStore

from weatheasy.build_copdem import _slopes
from weatheasy.correction import Source, apply, features, training
from weatheasy.correction.climatology import Climatology, build_climatology
from weatheasy.correction.precip import WET_FEAT, PrecipCorrector
from weatheasy.correction.sources import CFS2, ERA5
from weatheasy.correction.temperature import FEATURE_COLS, TemperatureCorrector
from weatheasy.error import CoordsError
from weatheasy.file_storage import FileStore, get_file_store
from weatheasy.query import Coords, get_cfs2_data, get_era5_data
from weatheasy.storage import get_store, open_dataset, prepare_dataset, write_batch
from weatheasy.util import get_storage


@pytest.fixture(name='stubbed_models')
def init_stubbed_models(monkeypatch: pytest.MonkeyPatch):
    class FakeTemperature:
        artifact_id = 'test-artifact'
        climatology = None

        def __init__(self, source: Source, _artifacts_dir, _dem_store) -> None:  # noqa: ANN001
            self.source = source

        def load_artifacts(self):
            return self

        def correct_grid(self, _day, values, _lats, _lons):  # noqa: ANN001
            return {'corrected': values - 2}

    class FakePrecipitation(FakeTemperature):
        def correct_grid(self, _day, values, _lats, _lons):  # noqa: ANN001
            return {'corrected': values * self.source.precip_to_mm / 2}

    monkeypatch.setattr(apply, 'TemperatureCorrector', FakeTemperature)
    monkeypatch.setattr(apply, 'PrecipCorrector', FakePrecipitation)


@pytest.fixture(name='temperature_correction')
def create_temperature_correction(tmp_path: Path):
    root, first = raw_store(tmp_path)
    lats, lons = ERA5.grid(root)
    corrector = TemperatureCorrector(
        ERA5, get_file_store(tmp_path), dem_store(tmp_path / 'dem.zarr', lats, lons)
    )
    corrector.model = Constant(1.5)
    corrector.grid_lats, corrector.grid_lons = lats, lons
    corrector.climatology = build_climatology(ERA5, root, ERA5.temp_var, 1.0)
    corrector.seasonal_bias = {m: np.ones((len(lats), len(lons)), np.float32) for m in range(1, 13)}
    return root, first, corrector


@pytest.fixture(name='cell_observations', params=[ERA5, CFS2], ids=['era5', 'cfs2'])
def create_cell_observations(request: pytest.FixtureRequest, tmp_path: Path):
    source = request.param
    root, station, target = heterogeneous_store(tmp_path, source)
    return source, root, station, target


@pytest.fixture(name='temperature_climatology')
def create_temperature_climatology(cell_observations: tuple):
    source, root, _, _ = cell_observations
    return build_climatology(source, root, 'TMP', 1.0)


@pytest.fixture(name='precipitation_climatology')
def create_precipitation_climatology(cell_observations: tuple):
    source, root, _, _ = cell_observations
    return build_climatology(source, root, 'PRATE', source.precip_to_mm, wet_threshold=1.0)


@pytest.fixture(name='cell_terrain')
def create_cell_terrain(cell_observations: tuple, tmp_path: Path):
    source, root, _, _ = cell_observations
    return dem_store(tmp_path / 'dem.zarr', *source.grid(root))


@pytest.fixture(name='recording_corrector', params=['temperature', 'precipitation'])
def create_recording_corrector(
    request: pytest.FixtureRequest, cell_observations: tuple, cell_terrain: Store, tmp_path: Path
):
    source, root, station, _ = cell_observations
    if request.param == 'temperature':
        corrector = TemperatureCorrector(source, get_file_store(tmp_path), cell_terrain)
        corrector.model = RecordTemperature()
        corrector.climatology = build_climatology(source, root, 'TMP', 1.0)
        corrector.seasonal_bias = {
            m: np.ones_like(corrector.climatology.mean[0]) for m in range(1, 13)
        }
    else:
        corrector = PrecipCorrector(source, get_file_store(tmp_path), cell_terrain)
        corrector.clf = corrector.fallback_clf = RecordWet()
        corrector.reg = corrector.fallback_reg = Constant(np.log1p(8))
        corrector.climatology = build_climatology(
            source, root, 'PRATE', source.precip_to_mm, wet_threshold=1.0
        )
        corrector.train_coords = station[['lat', 'lon']].to_numpy()
        corrector.seasonal = {
            'log_ratio': {m: np.zeros_like(corrector.climatology.mean[0]) for m in range(1, 13)},
            'wet_bias': {m: np.full_like(corrector.climatology.mean[0], 0.5) for m in range(1, 13)},
        }
    corrector.grid_lats, corrector.grid_lons = source.grid(root)
    return corrector


@pytest.fixture(name='extra_raw_day')
def create_extra_raw_day(cell_observations: tuple, temperature_climatology: Climatology):
    source, _, _, _ = cell_observations
    climate = temperature_climatology
    return prepare_dataset(
        xr.Dataset(
            {
                'TMP': (
                    ('time', 'latitude', 'longitude'),
                    np.full((1, *climate.mean.shape[1:]), 100.0, np.float32),
                    {'units': 'degC'},
                ),
                'PRATE': (
                    ('time', 'latitude', 'longitude'),
                    np.full((1, *climate.mean.shape[1:]), 100.0 / source.precip_to_mm, np.float32),
                    {'units': 'kg m-2 s-1'},
                ),
            },
            coords={
                'time': [np.datetime64('2020-01-04')],
                'latitude': climate.lats,
                'longitude': climate.lons,
            },
        ),
        crs='EPSG:4326',
    )


@pytest.fixture(name='station_weather')
def create_station_weather(tmp_path: Path):
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
    return root, first, station


@pytest.fixture(name='station_dem')
def create_station_dem(station_weather: tuple):
    root, _, _ = station_weather
    lats, lons = ERA5.grid(root)
    return {
        'lats': lats,
        'lons': lons,
        'elevation': np.full((len(lats), len(lons)), 150, np.float32),
        'slope_we': np.zeros((len(lats), len(lons)), np.float32),
        'slope_sn': np.zeros((len(lats), len(lons)), np.float32),
    }


@pytest.fixture(name='artifact_directory', params=['local', 'memory'])
def create_artifact_directory(request: pytest.FixtureRequest, tmp_path: Path) -> FileStore:
    if request.param == 'local':
        return get_file_store(tmp_path / 'models')
    return FileStore(MemoryFileSystem(), f'/models/{tmp_path.name}')


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
        root.attrs['updated'] = (start + timedelta(days=max(offsets) + 4)).isoformat()
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


@pytest.mark.usefixtures('stubbed_models')
def test_correction_append_and_rebuild_resets_progress(
    tmp_path: Path,
):
    root, first = raw_store(tmp_path)
    args = (ERA5, root, get_file_store(tmp_path / 'artifacts'), get_store(tmp_path / 'dem.zarr'))
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


@pytest.mark.usefixtures('stubbed_models')
def test_cfs_cropped_correction_has_own_grid(tmp_path: Path):
    root, first = raw_store(tmp_path, CFS2)
    assert (
        apply.correct_store(CFS2, root, get_file_store(tmp_path), get_store(tmp_path / 'dem.zarr'))
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


@pytest.mark.usefixtures('stubbed_models')
def test_cfs_correction_resumes_across_missing_calendar_day(tmp_path: Path):
    root, first = raw_store(tmp_path, CFS2, day_offsets=(0, 2))
    args = (CFS2, root, get_file_store(tmp_path), get_store(tmp_path / 'dem.zarr'))
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


@pytest.mark.usefixtures('stubbed_models')
def test_correction_rejects_skipping_prepared_day(tmp_path: Path):
    root, first = raw_store(tmp_path, days=3)
    args = (ERA5, root, get_file_store(tmp_path), get_store(tmp_path / 'dem.zarr'))
    assert apply.correct_store(*args, end=first) == 1
    with pytest.raises(ValueError, match='cannot skip prepared day 2020-01-02'):
        apply.correct_store(*args, begin=first + timedelta(days=2))


def test_missing_precipitation_stays_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, first = raw_store(tmp_path)
    lats, lons = ERA5.grid(root)
    corrector = PrecipCorrector(
        ERA5, get_file_store(tmp_path), dem_store(tmp_path / 'dem.zarr', lats, lons)
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


def test_temperature_corrector_applies_bias(temperature_correction: tuple):
    root, first, corrector = temperature_correction
    corrected = corrector.correct(root, first)
    np.testing.assert_allclose(corrected['original'] - corrected['corrected'], 1.5)


def test_temperature_corrector_suppresses_small_bias(temperature_correction: tuple):
    root, first, corrector = temperature_correction
    corrector.seasonal_bias[1][:] = 0.1
    corrected = corrector.correct(root, first)
    np.testing.assert_allclose(corrected['original'], corrected['corrected'])


def test_temperature_climatology_cell_mean(cell_observations: tuple):
    source, root, _, (li, lo) = cell_observations
    climate = build_climatology(source, root, 'TMP', 1.0)
    assert climate.mean[0, li, lo] == pytest.approx(10.0)


def test_precipitation_climatology_cell_mean(cell_observations: tuple):
    source, root, _, (li, lo) = cell_observations
    climate = build_climatology(source, root, 'PRATE', source.precip_to_mm, wet_threshold=1.0)
    assert climate.mean[0, li, lo] == pytest.approx(4.0)


def test_precipitation_climatology_wet_fraction(cell_observations: tuple):
    source, root, _, (li, lo) = cell_observations
    climate = build_climatology(source, root, 'PRATE', source.precip_to_mm, wet_threshold=1.0)
    assert climate.require_wet_fraction()[0, li, lo] == pytest.approx(0.5)


def test_temperature_climatology_preserves_missing_month(cell_observations: tuple):
    source, root, _, _ = cell_observations
    climate = build_climatology(source, root, 'TMP', 1.0)
    assert np.isnan(climate.mean[1]).all()


def test_training_temperature_anomaly(
    cell_observations: tuple, temperature_climatology: Climatology
):
    source, root, station, _ = cell_observations
    sampled = training.sample_reanalysis(source, root, station, 'TMP', 1.0)
    rows = training._temp_features(sampled, temperature_climatology)
    assert rows.era5_anomaly.iloc[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ('column', 'expected'), [('era5_precip_anom', 4.0), ('era5_wet_fraction', 0.5)]
)
def test_training_precipitation_features(
    cell_observations: tuple, precipitation_climatology: Climatology, column: str, expected: float
):
    source, root, station, _ = cell_observations
    sampled = training.sample_reanalysis(source, root, station, 'PRATE', source.precip_to_mm)
    rows = training._precip_features(sampled, precipitation_climatology)
    assert rows[column].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize('recording_corrector', ['temperature'], indirect=True)
def test_temperature_application_matches_training_features(
    cell_observations: tuple, recording_corrector: TemperatureCorrector
):
    source, root, station, _ = cell_observations
    corrector = recording_corrector
    assert corrector.climatology is not None
    rows = training._temp_features(
        training.sample_reanalysis(source, root, station, 'TMP', 1.0), corrector.climatology
    )
    corrector.correct(root, date(2020, 1, 2))
    assert corrector.model.captured[0, FEATURE_COLS.index('era5_anomaly')] == pytest.approx(
        rows.era5_anomaly.iloc[0]
    )


@pytest.mark.parametrize('column', ['era5_precip_anom', 'era5_wet_fraction'])
@pytest.mark.parametrize('recording_corrector', ['precipitation'], indirect=True)
def test_precipitation_application_matches_training_features(
    cell_observations: tuple, recording_corrector: PrecipCorrector, column: str
):
    source, root, station, _ = cell_observations
    corrector = recording_corrector
    assert corrector.climatology is not None
    rows = training._precip_features(
        training.sample_reanalysis(source, root, station, 'PRATE', source.precip_to_mm),
        corrector.climatology,
    )
    corrector.correct(root, date(2020, 1, 2))
    assert corrector.clf.captured[0, WET_FEAT.index(column)] == pytest.approx(rows[column].iloc[0])


@pytest.mark.parametrize('cell_observations', [ERA5], indirect=True, ids=['era5'])
@pytest.mark.parametrize('recording_corrector', ['temperature'], indirect=True)
def test_new_raw_days_do_not_change_frozen_temperature_features(
    cell_observations: tuple,
    recording_corrector: TemperatureCorrector,
    extra_raw_day: xr.Dataset,
):
    source, root, _, _ = cell_observations
    write_batch(root.store, source.path(root), extra_raw_day)
    corrector = recording_corrector
    corrector.correct(root, date(2020, 1, 2))
    assert corrector.model.captured[0, FEATURE_COLS.index('era5_anomaly')] == pytest.approx(3.0)


@pytest.mark.parametrize('cell_observations', [ERA5], indirect=True, ids=['era5'])
def test_correction_without_month_climatology_preserves_raw_and_returns_nan(
    cell_observations: tuple,
    extra_raw_day: xr.Dataset,
    recording_corrector: TemperatureCorrector | PrecipCorrector,
):
    source, root, _, _ = cell_observations
    corrector = recording_corrector
    february = extra_raw_day.assign_coords(time=[np.datetime64('2020-02-01')])
    write_batch(root.store, source.path(root), february)
    result = corrector.correct(root, date(2020, 2, 1))
    assert np.isnan(result['corrected']).all()
    assert np.isfinite(result['original']).all()


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
    loaded = training.prepare_stations(pd.read_csv(csv), ['precipitation'])
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


def test_station_terrain_enrichment(station_weather: tuple, station_dem: features.Dem):
    _, _, station = station_weather
    enriched = training.enrich_stations(station, station_dem)
    assert enriched['elevation'].iloc[0] == 150


def test_station_temperature_sampling(station_weather: tuple):
    root, _, station = station_weather
    sampled = training.sample_reanalysis(ERA5, root, station, 'TMP', 1.0)
    assert sampled['reanalysis'].iloc[0] == 10


def test_source_precipitation_units_and_grid(station_weather: tuple):
    root, first, _ = station_weather
    raw, lats, lons = ERA5.read_day(root, 'PRATE', first)
    assert lats[0] == pytest.approx(55)
    assert lons[-1] == pytest.approx(38)
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
    from weatheasy._cli import main

    root, first = raw_store(tmp_path)
    dem_path = tmp_path / 'dem.zarr'
    dem_store(dem_path, *ERA5.grid(root))
    stations = tmp_path / 'stations.parquet'
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
    ).to_parquet(stations, index=False, compression='zstd')
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
    main(['correction', 'train', *common, '--stations', str(stations)])
    main(['correction', 'apply', *common, '--end', first.isoformat()])
    old_id = root[ERA5.correction_path].attrs['correction_artifact_id']
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 1
        assert np.isfinite(ds.TMP_corr).all()
        assert float(ds.TMP_corr.mean()) < 10
        assert np.isnan(ds.PRATE_corr[0, 1, 1])
    main(['correction', 'train', *common, '--stations', str(stations), '--only', 'temp'])
    main(['correction', 'apply', *common])
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
            ERA5, root, get_file_store(artifacts), get_store(tmp_path / 'missing.zarr'), end=first
        )
    assert ERA5.correction_path not in root


def test_filesystem_model_training_application_and_resume(
    artifact_directory: FileStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, first = raw_store(tmp_path)
    terrain = dem_store(tmp_path / 'dem.zarr', *ERA5.grid(root))
    stations = pd.DataFrame(
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
    )
    monkeypatch.setitem(training._XGB_PARAMS, 'n_estimators', 8)
    original = stations.copy(deep=True)
    training.build_artifacts(ERA5, root, stations, terrain, artifact_directory)
    pd.testing.assert_frame_equal(stations, original)
    # A fresh reader has no handles or in-memory model state from training.
    fresh = FileStore(artifact_directory.fs, artifact_directory.root)
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
def test_manifest_validates_filesystem_bundle(artifact_directory: FileStore, damage: str):
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
    artifact_directory: FileStore, monkeypatch: pytest.MonkeyPatch
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


@pytest.mark.usefixtures('stubbed_models')
def test_correction_reads_each_source_chunk_once_across_cropped_time_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = MemoryStore()
    latitudes, longitudes = np.arange(40, 84), np.arange(18, 88)
    dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-05-10'))
    values = (
        np.arange(130)[:, None, None] * 1000
        + np.arange(44)[None, :, None] * 10
        + np.arange(70)[None, None, :]
    ).astype(np.float32)
    values[4, 10, 10] = np.nan
    raw = xr.Dataset(
        {
            'TMP': (('time', 'latitude', 'longitude'), values, {'units': 'degC'}),
            'PRATE': (
                ('time', 'latitude', 'longitude'),
                values / 21600,
                {'units': 'kg m-2 s-1'},
            ),
        },
        coords={'time': dates, 'latitude': latitudes, 'longitude': longitudes},
    )
    write_batch(store, 'reanalysis/flx', prepare_dataset(raw, crs='EPSG:4326'))
    root = zarr.open_group(store, use_consolidated=False)
    counts: Counter[str] = Counter()
    original_get = MemoryStore.get
    original_get_sync = MemoryStore.get_sync

    def record(key: str) -> None:
        if key.startswith(('reanalysis/flx/TMP/c/', 'reanalysis/flx/PRATE/c/')):
            counts[key] += 1

    async def counted_get(self: MemoryStore, key: str, *args, **kwargs):
        record(key)
        return await original_get(self, key, *args, **kwargs)

    def counted_get_sync(self: MemoryStore, key: str, *args, **kwargs):
        record(key)
        return original_get_sync(self, key, *args, **kwargs)

    # Instrument the class so stores cloned by xarray retain the read counters.
    monkeypatch.setattr(MemoryStore, 'get', counted_get)
    monkeypatch.setattr(MemoryStore, 'get_sync', counted_get_sync)
    assert (
        apply.correct_store(
            CFS2, root, get_file_store(tmp_path / 'artifacts'), get_store(tmp_path / 'dem.zarr')
        )
        == 130
    )
    expected_reads = {
        f'reanalysis/flx/{var}/c/{time}/{lat}/{lon}': 1
        for var in ('TMP', 'PRATE')
        for time in (0, 1)
        for lat in (0, 1)
        for lon in (0, 1, 2)
    }
    assert counts == expected_reads
    with open_dataset(store, 'reanalysis/correction') as result:
        np.testing.assert_array_equal(result.time, dates.astype('datetime64[ns]'))
        np.testing.assert_array_equal(result.latitude, latitudes[1:43])
        np.testing.assert_array_equal(result.longitude, longitudes[1:])
        np.testing.assert_array_equal(result.TMP_corr, values[:, 1:43, 1:] - 2)
        np.testing.assert_allclose(result.PRATE_corr, values[:, 1:43, 1:] / 2, rtol=1e-6)
