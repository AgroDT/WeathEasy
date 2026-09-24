"""Offline correction on coordinate-aware Zarr groups."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weatheasy import Coords, get_cfs2_data, get_era5_data
from weatheasy.build_copdem import _slopes
from weatheasy.correction import apply, features, training
from weatheasy.correction.precip import PrecipCorrector
from weatheasy.correction.sources import CFS2, ERA5
from weatheasy.correction.temperature import TemperatureCorrector
from weatheasy.error import CoordsError
from weatheasy.storage import open_dataset, prepare_dataset, write_batch
from weatheasy.util import get_storage


class Constant:
    def __init__(self, value) -> None:
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, np.float32)


class AlwaysWet:
    def predict_proba(self, x):
        return np.tile([0.1, 0.9], (len(x), 1))


def raw_store(tmp_path, source=ERA5, days=2, day_offsets=None, lon_start=37):
    root = get_storage(str(tmp_path / 'data'))
    if source == ERA5:
        lats, lons = 55 + np.arange(11) * 0.1, lon_start + np.arange(11) * 0.1
    else:
        lats, lons = np.arange(40, 84), np.arange(18, 182)
    start = date(2020, 1, 1)
    offsets = tuple(range(days)) if day_offsets is None else day_offsets
    shape = (len(offsets), len(lats), len(lons))
    tmp = np.full(shape, 10, np.float32)
    prate = np.full(shape, 10 / source.precip_to_mm, np.float32)
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


def stub_models(monkeypatch):
    class FakeTemperature:
        def __init__(self, source, _artifacts_dir, _dem_path) -> None:
            self.source = source

        def load_artifacts(self):
            return self

        def correct(self, root, day):
            values, _, _ = self.source.read_day(root, 'TMP', day)
            return {'corrected': values - 2}

    class FakePrecipitation(FakeTemperature):
        def correct(self, root, day):
            values, _, _ = self.source.read_day(root, 'PRATE', day)
            return {'corrected': values * self.source.precip_to_mm / 2}

    monkeypatch.setattr(apply, 'TemperatureCorrector', FakeTemperature)
    monkeypatch.setattr(apply, 'PrecipCorrector', FakePrecipitation)


def test_correction_append_and_rebuild_resets_progress(tmp_path, monkeypatch):
    root, first = raw_store(tmp_path)
    stub_models(monkeypatch)
    args = (ERA5, root, tmp_path / 'artifacts', tmp_path / 'dem.nc')
    assert apply.correct_store(*args, end=first) == 1
    assert apply.correct_store(*args) == 1
    assert apply.correct_store(*args) == 0
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 2
        assert ds.PRATE_corr.attrs['units'] == 'mm/day'
    result = get_era5_data(
        root=root,
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


def test_cfs_cropped_correction_has_own_grid(tmp_path, monkeypatch):
    root, first = raw_store(tmp_path, CFS2)
    stub_models(monkeypatch)
    assert apply.correct_store(CFS2, root, tmp_path, tmp_path / 'dem.nc') == 2
    inside = get_cfs2_data(
        root=root,
        begin=first,
        end=first,
        coords=Coords(55, 37),
        variables=['TMP', 'TMP_corr', 'PRATE_corr'],
    )
    np.testing.assert_allclose(inside[:, 0], [10, 8, 5])
    outside = get_cfs2_data(
        root=root,
        begin=first,
        end=first,
        coords=Coords(40, 18),
        variables=['TMP'],
    )
    np.testing.assert_allclose(outside, [[10]])
    with pytest.raises(CoordsError):
        get_cfs2_data(
            root=root,
            begin=first,
            end=first,
            coords=Coords(40, 18),
            variables=['TMP_corr'],
        )


def test_cfs_correction_resumes_across_missing_calendar_day(tmp_path, monkeypatch):
    root, first = raw_store(tmp_path, CFS2, day_offsets=(0, 2))
    stub_models(monkeypatch)
    args = (CFS2, root, tmp_path, tmp_path / 'dem.nc')
    assert apply.correct_store(*args, end=first) == 1
    assert apply.correct_store(*args) == 1
    assert apply.correct_store(*args) == 0
    with open_dataset(root.store, CFS2.corrected_path(root)) as ds:
        np.testing.assert_array_equal(
            ds.time.to_numpy().astype('datetime64[D]'),
            [np.datetime64(first), np.datetime64(first + timedelta(days=2))],
        )
    result = get_cfs2_data(
        root=root,
        begin=first,
        end=first + timedelta(days=2),
        coords=Coords(55, 37),
        variables=['TMP', 'TMP_corr', 'PRATE_corr'],
    )
    np.testing.assert_allclose(result[:, [0, 2]], [[10, 10], [8, 8], [5, 5]])
    assert np.isnan(result[:, 1]).all()


def test_correction_rejects_skipping_prepared_day(tmp_path, monkeypatch):
    root, first = raw_store(tmp_path, days=3)
    stub_models(monkeypatch)
    args = (ERA5, root, tmp_path, tmp_path / 'dem.nc')
    assert apply.correct_store(*args, end=first) == 1
    with pytest.raises(ValueError, match='cannot skip prepared day 2020-01-02'):
        apply.correct_store(*args, begin=first + timedelta(days=2))


def test_missing_precipitation_stays_missing(tmp_path):
    root, first = raw_store(tmp_path)
    corrector = PrecipCorrector(ERA5, tmp_path, tmp_path / 'missing.nc')
    corrector.clf = corrector.fallback_clf = AlwaysWet()
    corrector.reg = corrector.fallback_reg = Constant(np.log1p(8))
    corrector.train_coords = np.array([[55.5, 37.5]], np.float32)
    lats, lons = ERA5.grid(root)
    corrector.grid_lats, corrector.grid_lons = lats, lons
    corrector.seasonal = {
        'log_ratio': {m: np.zeros((len(lats), len(lons)), np.float32) for m in range(1, 13)},
        'wet_bias': {m: np.full((len(lats), len(lons)), 0.5, np.float32) for m in range(1, 13)},
    }
    result = corrector.correct(root, first)
    assert np.isnan(result['corrected'][1, 1])
    assert result['corrected'][0, 0] == pytest.approx(8)
    # The missing cell is excluded from the spatial wet fraction, not counted dry.
    assert result['original'][0, 0] == pytest.approx(10)


def test_temperature_corrector_applies_and_suppresses_small_bias(tmp_path):
    root, first = raw_store(tmp_path)
    lats, lons = ERA5.grid(root)
    corrector = TemperatureCorrector(ERA5, tmp_path, tmp_path / 'missing.nc')
    corrector.model = Constant(1.5)
    corrector.grid_lats, corrector.grid_lons = lats, lons
    corrector.seasonal_bias = {m: np.ones((len(lats), len(lons)), np.float32) for m in range(1, 13)}
    corrected = corrector.correct(root, first)
    np.testing.assert_allclose(corrected['original'] - corrected['corrected'], 1.5)
    corrector.seasonal_bias[1][:] = 0.1
    suppressed = corrector.correct(root, first)
    np.testing.assert_allclose(suppressed['original'], suppressed['corrected'])


def test_training_discards_missing_observation_and_reanalysis(tmp_path):
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


def test_training_reports_and_skips_stations_outside_cell_bounds(tmp_path, caplog):
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


def test_station_terrain_enrichment_and_source_units(tmp_path):
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
    dem = {
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


def test_separate_training_and_apply_commands(tmp_path, monkeypatch):
    from weatheasy.correction.__main__ import main

    root, first = raw_store(tmp_path)
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
    monkeypatch.setitem(training._XGB_PARAMS, 'n_estimators', 8)  # noqa: SLF001
    common = [
        'era5',
        '--store',
        str(tmp_path / 'data'),
        '--dem',
        str(tmp_path / 'missing.nc'),
        '--artifacts',
        str(tmp_path / 'artifacts'),
    ]
    main(['train', *common, '--stations', str(stations)])
    main(['apply', *common, '--end', first.isoformat()])
    with open_dataset(root.store, ERA5.corrected_path(root)) as ds:
        assert ds.sizes['time'] == 1
        assert np.isfinite(ds.TMP_corr).all()
        assert float(ds.TMP_corr.mean()) < 10
        assert np.isnan(ds.PRATE_corr[0, 1, 1])
