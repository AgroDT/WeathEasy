"""Small offline EMDNA sequences with real CSV, daily NetCDF, and NPZ files."""

import copy
from datetime import date
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from weatheasy.emdna import PrecipitationProcessor, TemperatureProcessor, __main__ as cli
from weatheasy.emdna._base import EMDNABase
from weatheasy.emdna._inputs import process_period as process_file_period, read_stations
from weatheasy.file_storage import FileStore, get_file_store


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
    return kind(
        pd.read_csv(stations),
        get_file_store(output),
        min_stations=3,
        max_stations=4,
        n_ensemble=2,
        **kwargs,
    )


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

    def failed_save(stream: BinaryIO, **_payload):
        stream.write(b'incomplete archive')
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


def daily_dataset(model: EMDNABase, day: str) -> xr.Dataset:
    values = np.full((1, 2, 2), 20.0, np.float32)
    rain = np.full((1, 2, 2), 0.008, np.float32)
    return xr.Dataset(
        {
            't2m_mean': (('date', 'latitude', 'longitude'), values),
            'tp': (('date', 'latitude', 'longitude'), rain),
        },
        coords={'date': [np.datetime64(day)], 'latitude': model.lats, 'longitude': model.lons},
    )


def daily_netcdf(path: Path, model: EMDNABase, day: str):
    daily_dataset(model, day).to_netcdf(path)


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
    model = processor(kind, stations, tmp_path / 'out')
    model.era5 = daily_dataset(model, '2026-01-01')
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
    model = processor(PrecipitationProcessor, stations, tmp_path / 'out')
    model.era5 = daily_dataset(model, '2026-01-01')
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
        pd.read_csv(stations),
        get_file_store(tmp_path / 'out'),
        min_stations=3,
        max_stations=4,
        n_ensemble=3,
    )
    with pytest.raises(ValueError, match='Incompatible ensemble or grid'):
        incompatible.process_period()


@pytest.mark.parametrize(
    ('kind', 'prefix'), [(TemperatureProcessor, 'temp'), (PrecipitationProcessor, 'precip')]
)
def test_remote_inputs_restart_and_failed_publish(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[EMDNABase],
    prefix: str,
):
    fs = MemoryFileSystem()
    inputs = FileStore(fs, f'/emdna-{prefix}-inputs')
    output = FileStore(fs, f'/emdna-{prefix}-results')
    with inputs.open('stations.csv', 'wb') as stream:
        stream.write(stations.read_bytes())
    local = processor(kind, stations, tmp_path / 'full')
    daily_netcdf(tmp_path / '202601_daily.nc', local, '2026-01-01')
    with inputs.open('202601_daily.nc', 'wb') as stream:
        stream.write((tmp_path / '202601_daily.nc').read_bytes())
    local.era5 = daily_dataset(local, '2026-01-01')
    local._rng = np.random.default_rng(42)
    assert [r['status'] for r in local.process_period()] == ['success'] * 3

    def remote_processor():
        return kind(
            read_stations(inputs.child('stations.csv')),
            output,
            min_stations=3,
            max_stations=4,
            n_ensemble=2,
        )

    downloads = []
    original_open = fs.open

    def count_reads(path: str, mode: str = 'rb', **kwargs):
        if path.endswith('.nc') and mode == 'rb':
            downloads.append(path)
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(fs, 'open', count_reads)
    partial = remote_processor()
    partial._rng = np.random.default_rng(42)
    assert process_file_period(partial, inputs, end=date(2026, 1, 1))[0]['status'] == 'success'
    resumed = remote_processor()
    # Serialization fails after writing bytes: no incomplete day may become discoverable.
    original = np.savez_compressed

    def failed_save(stream: BinaryIO, **_payload):
        stream.write(b'incomplete archive')
        msg = 'simulated storage failure'
        raise OSError(msg)

    monkeypatch.setattr(np, 'savez_compressed', failed_save)
    assert process_file_period(resumed, inputs)[0]['status'] == 'error'
    assert not output.exists(f'emdna_{prefix}_20260102.npz')
    monkeypatch.setattr(np, 'savez_compressed', original)
    downloads.clear()
    assert [r['status'] for r in process_file_period(resumed, inputs)] == ['success'] * 2
    assert downloads == [inputs.child('202601_daily.nc').root]
    assert resumed.era5 is None
    downloads.clear()
    assert process_file_period(remote_processor(), inputs) == []
    assert downloads == []
    for day in (1, 2, 3):
        name = f'emdna_{prefix}_2026010{day}.npz'
        with (
            np.load(tmp_path / 'full' / name) as expected,
            output.open(name) as stream,
            np.load(stream) as actual,
        ):
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])


def test_cli_preserves_s3_uris_and_shares_filesystem(monkeypatch: pytest.MonkeyPatch):
    fs = MemoryFileSystem()
    resolutions = []
    captured = {}
    with fs.open('bucket/stations.csv', 'wb') as stream:
        stream.write(b'date\n2026-01-01\n')

    def resolve(uri: str, *, filesystem: AbstractFileSystem | None = None):
        resolutions.append((uri, filesystem))
        return FileStore(fs, uri[5:])

    class Processor:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.stations = kwargs['stations'].copy()
            self.stations['date'] = pd.to_datetime(self.stations['date'])

        def pending_dates(self, _start: date | None, _end: date | None):
            return []

        def process_period(self, _start: date | None, _end: date | None):
            return []

    monkeypatch.setattr(cli, 'get_file_store', resolve)
    monkeypatch.setattr('weatheasy.emdna.temperature.TemperatureProcessor', Processor)
    monkeypatch.setattr(
        'sys.argv',
        [
            'weatheasy.emdna',
            'temperature',
            's3://bucket/stations.csv',
            '-o',
            's3://bucket/results',
            '--era5',
            's3://bucket/era5',
        ],
    )
    cli.main(configure_logging=False)
    assert resolutions == [
        ('s3://bucket/stations.csv', None),
        ('s3://bucket/stations.csv', fs),
        ('s3://bucket/results', fs),
        ('s3://bucket/era5', fs),
    ]
    pd.testing.assert_frame_equal(captured['stations'], pd.DataFrame({'date': ['2026-01-01']}))
    assert captured['output_dir'].root == 'bucket/results'
    assert captured['output_dir'].fs is fs


@pytest.mark.parametrize('kind', [TemperatureProcessor, PrecipitationProcessor])
def test_in_memory_inputs_are_not_mutated_or_read_from_files(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[EMDNABase],
):
    observations = pd.read_csv(stations)
    observations['date'] = pd.to_datetime(observations[['year', 'month', 'day']])
    observations = observations.drop(columns=['year', 'month', 'day'])
    expected = observations.copy(deep=True)
    grid = processor(kind, stations, tmp_path / 'grid')
    era5 = daily_dataset(grid, '2026-01-01')
    expected_era5 = era5.copy(deep=True)

    def unexpected_read(*_args, **_kwargs):
        pytest.fail('Library processing must use the prepared in-memory inputs')

    monkeypatch.setattr(pd, 'read_csv', unexpected_read)
    monkeypatch.setattr(xr, 'open_dataset', unexpected_read)
    model = kind(
        observations,
        get_file_store(tmp_path / 'out'),
        era5=era5,
        min_stations=3,
        max_stations=4,
        n_ensemble=2,
    )
    assert [result['status'] for result in model.process_period()] == ['success'] * 3
    pd.testing.assert_frame_equal(observations, expected)
    xr.testing.assert_identical(era5, expected_era5)


def test_cli_monthly_batches_stop_after_failure_and_restart(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = pd.read_csv(stations)
    february = frame[frame.day == 1].copy()
    february['month'] = 2
    frame = pd.concat([frame, february], ignore_index=True)
    frame = frame[~((frame.month == 1) & (frame.day == 2) & (frame.station_id > 0))]
    frame.to_csv(stations, index=False)
    monkeypatch.setattr(
        'sys.argv',
        [
            'weatheasy.emdna',
            'temperature',
            str(stations),
            '-o',
            str(tmp_path / 'out'),
            '--n-ensemble',
            '2',
            '--min-stations',
            '3',
            '--max-stations',
            '4',
        ],
    )
    with pytest.raises(SystemExit) as failure:
        cli.main(configure_logging=False)
    assert failure.value.code == 1
    assert not (tmp_path / 'out' / 'emdna_temp_20260201.npz').exists()
    complete = pd.read_csv(stations)
    missing = complete[
        (complete.month == 1) & (complete.day == 1) & (complete.station_id > 0)
    ].copy()
    missing['day'] = 2
    pd.concat([complete, missing]).to_csv(stations, index=False)
    cli.main(configure_logging=False)
    assert (tmp_path / 'out' / 'emdna_temp_20260201.npz').exists()


def test_monthly_file_adapter_uses_each_dataset_and_closes_it(
    stations: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    observations = pd.read_csv(stations)
    january = observations[observations.day == 1].copy()
    february = observations[observations.day <= 2].copy()
    february['month'] = 2
    observations = pd.concat([january, february], ignore_index=True)
    model = TemperatureProcessor(
        observations, get_file_store(tmp_path / 'out'), min_stations=3, max_stations=4, n_ensemble=2
    )
    fs = MemoryFileSystem()
    directory = FileStore(fs, '/emdna-two-month-inputs')
    for month, temperature in [('202601', 20), ('202602', 40)]:
        dataset = daily_dataset(model, f'{month[:4]}-{month[4:]}-01')
        dataset['t2m_mean'][:] = temperature
        local = tmp_path / f'{month}_daily.nc'
        dataset.to_netcdf(local)
        with directory.open(local.name, 'wb') as stream:
            stream.write(local.read_bytes())
    opened = []
    original = xr.open_dataset

    def tracked_open(path: str):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(xr, 'open_dataset', tracked_open)
    assert [result['status'] for result in process_file_period(model, directory)] == ['success'] * 3
    assert len(opened) == 2
    assert all(not Path(path).exists() for path in opened)
    for month, temperature in [('202601', 20), ('202602', 40)]:
        with np.load(tmp_path / 'out' / f'emdna_temp_{month}01.npz') as result:
            np.testing.assert_allclose(
                result['temperature'], 0.7 * result['temperature_cai'] + 0.3 * temperature
            )
    with np.load(tmp_path / 'out' / 'emdna_temp_20260202.npz') as result:
        np.testing.assert_array_equal(result['temperature'], result['temperature_cai'])
    assert process_file_period(model, directory) == []
    assert len(opened) == 2
