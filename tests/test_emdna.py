"""Small offline EMDNA sequences with real CSV, daily NetCDF, and NPZ files."""

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weatheasy.emdna import PrecipitationProcessor, TemperatureProcessor, __main__ as cli
from weatheasy.emdna._base import EMDNABase


@pytest.fixture
def stations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def small_grid(self):  # noqa: ANN001
        self.lats = np.array([50.0, 50.1], dtype=np.float32)
        self.lons = np.array([30.0, 30.1], dtype=np.float32)
        self.grid_lons, self.grid_lats = np.meshgrid(self.lons, self.lats)

    monkeypatch.setattr(EMDNABase, '_setup_grid', small_grid)
    rows = []
    for day in (1, 2, 3):
        for sid, (lat, lon) in enumerate(((50, 30), (50, 30.1), (50.1, 30), (50.1, 30.1))):
            rows.append(
                {
                    'station_id': sid,
                    'lat': lat,
                    'lon': lon,
                    'year': 2026,
                    'month': 1,
                    'day': day,
                    'temperature': 10 + day + sid,
                    'precipitation': day + sid,
                }
            )
    path = tmp_path / 'stations.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def processor(kind: type[EMDNABase], stations: Path, output: Path, **kwargs):
    return kind(stations, output, min_stations=3, max_stations=4, n_ensemble=2, **kwargs)


def test_distances_and_station_selection(stations: Path, tmp_path: Path):
    coords = np.array([[60, 2], [61, 0], [60, 0]], dtype=float)
    np.testing.assert_allclose(EMDNABase._distances(np.array([60, 0]), coords), [111, 111, 0])
    np.testing.assert_allclose(EMDNABase._distances(np.array([0, 0]), np.array([[0, 2]])), [222])
    np.testing.assert_allclose(
        EMDNABase._distances(np.array([60, 179]), np.array([[60, -179]])), [111]
    )
    model = processor(TemperatureProcessor, stations, tmp_path / 'out')
    model.min_stations = 1
    model.max_stations = 1
    model.search_radius_km = 150
    # Correct longitude scaling makes this station nearer than a 1.5-degree latitude offset.
    assert model._select_stations(np.array([60, 0]), np.array([[60, 2], [61.5, 0]])).tolist() == [
        True,
        False,
    ]


def test_temperature_member_state(stations: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = processor(TemperatureProcessor, stations, tmp_path / 'out')
    fields = iter(np.full((2, 2), value, np.float32) for value in (1, 10, 2, 20))
    monkeypatch.setattr(model, '_spatial_random_field', lambda _: next(fields))
    zero, one = np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)
    first = model._generate_ensemble(zero, one)
    second = model._generate_ensemble(zero, one)
    np.testing.assert_allclose(first[:, 0, 0], [1, 10])
    np.testing.assert_allclose(second[:, 0, 0], [0.8 * 1 + 0.6 * 2, 0.8 * 10 + 0.6 * 20])


def test_precipitation_member_state(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = processor(PrecipitationProcessor, stations, tmp_path / 'out')
    fields = iter(np.full((2, 2), value, np.float32) for value in (1, 10, 2, 20))
    monkeypatch.setattr(model, '_spatial_random_field', lambda _: next(fields))
    one = np.ones((2, 2), np.float32)
    zero = np.zeros((2, 2), np.float32)
    model._generate_ensemble(one, one, one, zero)
    np.testing.assert_allclose(
        [model._previous_scrf[i][0, 0] for i in (0, 1)], np.sqrt(0.91) * np.array([1, 10])
    )
    model._generate_ensemble(one, one, one, zero)
    # Preserve the donor cross-correlation formula, but use each member's own prior state.
    np.testing.assert_allclose(
        [model._previous_scrf[i][0, 0] for i in (0, 1)], 0.91 * np.array([1, 10])
    )


@pytest.mark.parametrize(
    ('kind', 'prefix'), [(TemperatureProcessor, 'temp'), (PrecipitationProcessor, 'precip')]
)
def test_resume_matches_uninterrupted_and_new_dates(
    stations: Path,
    tmp_path: Path,
    kind: type[EMDNABase],
    prefix: str,
):
    uninterrupted = processor(kind, stations, tmp_path / 'full')
    uninterrupted._rng = np.random.default_rng(42)
    assert [r['status'] for r in uninterrupted.process_period()] == ['success'] * 3
    partial = processor(kind, stations, tmp_path / 'resume')
    partial._rng = np.random.default_rng(42)
    assert partial.process_period(end=date(2026, 1, 1))[0]['status'] == 'success'
    resumed = processor(kind, stations, tmp_path / 'resume')
    assert len(resumed.process_period(end=date(2026, 1, 2))) == 1
    assert len(resumed.process_period()) == 1
    assert resumed.process_period() == []
    for day in (1, 2, 3):
        name = f'emdna_{prefix}_2026010{day}.npz'
        with (
            np.load(tmp_path / 'full' / name) as expected,
            np.load(tmp_path / 'resume' / name) as actual,
        ):
            for key in ('ensemble', 'scrf_state', 'rng_state'):
                np.testing.assert_array_equal(actual[key], expected[key])
    # A new CSV date is discovered automatically on the next invocation.
    frame = pd.read_csv(stations)
    extra = frame[frame.day == 3].copy()
    extra['day'] = 4
    pd.concat([frame, extra]).to_csv(stations, index=False)
    newer = processor(kind, stations, tmp_path / 'resume')
    assert len(newer.process_period()) == 1
    assert (tmp_path / 'resume' / f'emdna_{prefix}_20260104.npz').exists()


def test_partial_write_is_retryable_with_unchanged_state(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = processor(TemperatureProcessor, stations, tmp_path / 'out')
    model._rng = np.random.default_rng(42)
    model.process_period(end=date(2026, 1, 1))
    old_fields = copy.deepcopy(model._previous_scrf)
    old_rng = copy.deepcopy(model._rng.bit_generator.state)
    original = np.savez_compressed

    def failed_save(path: Path, **_payload):
        path.write_bytes(b'incomplete archive')
        msg = 'simulated storage failure'
        raise OSError(msg)

    monkeypatch.setattr(np, 'savez_compressed', failed_save)
    result = model.process_period()
    assert [r['status'] for r in result] == ['error']
    assert len(list((tmp_path / 'out').glob('*.npz'))) == 1
    assert model._rng.bit_generator.state == old_rng
    for member, field in old_fields.items():
        np.testing.assert_array_equal(model._previous_scrf[member], field)
    monkeypatch.setattr(np, 'savez_compressed', original)
    assert len(model.process_period()) == 2
    restarted = processor(TemperatureProcessor, stations, tmp_path / 'out')
    assert restarted.process_period() == []


def daily_netcdf(path: Path, model: EMDNABase, day: str):
    values = np.full((1, 2, 2), 20.0, np.float32)
    rain = np.full((1, 2, 2), 0.008, np.float32)
    xr.Dataset(
        {
            't2m_mean': (('date', 'latitude', 'longitude'), values),
            'tp': (('date', 'latitude', 'longitude'), rain),
        },
        coords={'date': [np.datetime64(day)], 'latitude': model.lats, 'longitude': model.lons},
    ).to_netcdf(path)


@pytest.mark.parametrize(
    ('kind', 'prefix'), [(TemperatureProcessor, 'temp'), (PrecipitationProcessor, 'precip')]
)
def test_daily_era5_exact_date_and_missing_date(
    stations: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind: type[EMDNABase],
    prefix: str,
):
    era5 = tmp_path / 'era5'
    era5.mkdir()
    model = processor(kind, stations, tmp_path / 'out', era5_dir=era5)
    daily_netcdf(era5 / '202601_daily.nc', model, '2026-01-01')
    assert model.process_day('2026-01-01')['status'] == 'success'
    with np.load(tmp_path / 'out' / f'emdna_{prefix}_20260101.npz') as result:
        if prefix == 'temp':
            np.testing.assert_allclose(
                result['temperature'], 0.7 * result['temperature_cai'] + 0.3 * 20
            )
        else:
            np.testing.assert_allclose(
                result['precipitation'], 0.75 * result['precipitation_cai'] + 0.25 * 8
            )
    assert model.process_day('2026-01-02')['status'] == 'success'
    assert 'ERA5 load failed for 2026-01-02' in caplog.text
    with np.load(tmp_path / 'out' / f'emdna_{prefix}_20260102.npz') as result:
        field = 'temperature' if prefix == 'temp' else 'precipitation'
        np.testing.assert_array_equal(result[field], result[f'{field}_cai'])


def test_missing_era5_cells_keep_station_precipitation_and_pop(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    era5 = tmp_path / 'era5'
    era5.mkdir()
    model = processor(PrecipitationProcessor, stations, tmp_path / 'out', era5_dir=era5)
    daily_netcdf(era5 / '202601_daily.nc', model, '2026-01-01')
    # Exercise NaN handling after interpolation, independently of xarray's NaN spread rules.
    rain = np.array([[np.nan, 8], [8, np.nan]], np.float32)
    real_loader = model._load_era5
    monkeypatch.setattr(xr.DataArray, 'interp', lambda _self, **_kwargs: xr.DataArray(rain))
    loaded, pop = real_loader('2026-01-01')
    np.testing.assert_array_equal(np.isnan(pop), np.isnan(rain))
    monkeypatch.setattr(model, '_load_era5', lambda _: (loaded, pop))
    one = np.ones((2, 2), np.float32)
    monkeypatch.setattr(model, '_cai', lambda *_: (10 * one, 0.8 * one, one, one, one))
    assert model.process_day('2026-01-01')['status'] == 'success'
    with np.load(tmp_path / 'out' / 'emdna_precip_20260101.npz') as result:
        np.testing.assert_allclose(result['precipitation'], [[10, 9.5], [9.5, 10]])
        np.testing.assert_allclose(result['pop'], [[0.8, 0.85], [0.85, 0.8]])


@pytest.mark.parametrize(('kind', 'prefix'), [('temperature', 'temp'), ('precipitation', 'precip')])
def test_cli_discovers_dates_and_noops(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    prefix: str,
):
    output = tmp_path / 'cli'
    monkeypatch.setattr(
        'sys.argv',
        [
            'weatheasy.emdna',
            kind,
            str(stations),
            '-o',
            str(output),
            '--n-ensemble',
            '2',
            '--min-stations',
            '3',
            '--max-stations',
            '4',
        ],
    )
    cli.main(configure_logging=False)
    paths = list(output.glob(f'emdna_{prefix}_*.npz'))
    assert len(paths) == 3
    original = {path: path.read_bytes() for path in paths}
    cli.main(configure_logging=False)
    assert all(path.read_bytes() == content for path, content in original.items())


def test_precipitation_cross_state_survives_restart(stations: Path, tmp_path: Path):
    trange = np.ones((2, 2), np.float32)
    full = processor(PrecipitationProcessor, stations, tmp_path / 'full')
    full._rng = np.random.default_rng(17)
    partial = processor(PrecipitationProcessor, stations, tmp_path / 'resume')
    partial._rng = np.random.default_rng(17)
    for model in (full, partial):
        assert model.process_day('2026-01-01', trange_scrf=trange)['status'] == 'success'
    full.process_day('2026-01-02', trange_scrf=trange)
    resumed = processor(PrecipitationProcessor, stations, tmp_path / 'resume')
    resumed.process_day('2026-01-02', trange_scrf=trange)
    with (
        np.load(tmp_path / 'full' / 'emdna_precip_20260102.npz') as expected,
        np.load(tmp_path / 'resume' / 'emdna_precip_20260102.npz') as actual,
    ):
        np.testing.assert_array_equal(actual['scrf_state'], expected['scrf_state'])
        np.testing.assert_array_equal(actual['ensemble'], expected['ensemble'])


def test_insufficient_day_does_not_advance_progress(stations: Path, tmp_path: Path):
    frame = pd.read_csv(stations)
    frame = frame[~((frame.day == 2) & (frame.station_id > 0))]
    frame.to_csv(stations, index=False)
    model = processor(TemperatureProcessor, stations, tmp_path / 'out')
    assert [result['status'] for result in model.process_period()] == [
        'success',
        'insufficient_stations',
    ]
    assert not (tmp_path / 'out' / 'emdna_temp_20260103.npz').exists()
    assert [result['status'] for result in model.process_period()] == ['insufficient_stations']


def test_resume_rejects_different_ensemble(stations: Path, tmp_path: Path):
    model = processor(TemperatureProcessor, stations, tmp_path / 'out')
    model.process_period(end=date(2026, 1, 1))
    incompatible = TemperatureProcessor(
        stations, tmp_path / 'out', min_stations=3, max_stations=4, n_ensemble=3
    )
    with pytest.raises(ValueError, match='Incompatible ensemble or grid'):
        incompatible.process_period()
