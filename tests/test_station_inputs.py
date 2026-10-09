from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from weatheasy._cli._inputs import read_stations
from weatheasy._cli.commands import correction_train
from weatheasy.file_storage import FileStore, get_file_store


@pytest.mark.parametrize('remote', [False, True])
def test_parquet_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, remote: bool):
    station_file = (
        FileStore(MemoryFileSystem(), '/projection/stations.parquet')
        if remote
        else get_file_store(tmp_path / 'stations.parquet')
    )
    observations = pd.DataFrame(
        {'station_id': ['20046'], 'temperature': [-24.5], 'elevation': [21.0], 'unused': ['text']}
    )
    with station_file.open(mode='wb') as stream:
        observations.to_parquet(stream, index=False, compression='zstd')

    original = pd.read_parquet
    requested = []

    def recording_read(*args, **kwargs):
        requested.append(kwargs.get('columns'))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', recording_read)
    selected = read_stations(station_file, columns=['station_id', 'temperature'])
    pd.testing.assert_frame_equal(
        selected, pd.DataFrame({'station_id': ['20046'], 'temperature': [-24.5]})
    )
    assert requested == [['station_id', 'temperature']]
    pd.testing.assert_frame_equal(read_stations(station_file), observations)


@pytest.mark.parametrize(
    ('only', 'values'),
    [
        (None, ['temperature', 'precipitation']),
        ('temp', ['temperature']),
        ('precip', ['precipitation']),
    ],
)
def test_training_reads_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, only: str | None, values: list[str]
):
    path = tmp_path / 'stations.parquet'
    observations = pd.DataFrame(
        {'station_id': ['20046'], 'lat': [80.6], 'lon': [58.0], 'date': ['1980-01-01']}
    )
    for column in values:
        observations[column] = -24.5 if column == 'temperature' else 1.7
    observations['unused'] = 'text'
    observations.to_parquet(path, index=False, compression='zstd')
    storage = SimpleNamespace(
        group=lambda _: None,
        store=lambda _: None,
        files=get_file_store,
    )
    monkeypatch.setattr(correction_train, 'Storage', lambda *_: storage)
    captured = []

    def build_artifacts(
        _source: object,
        _weather: object,
        stations: pd.DataFrame,
        _dem: object,
        _artifacts: object,
        *,
        only: str | None,
    ):
        captured.append((stations, only))

    monkeypatch.setattr(correction_train.training, 'build_artifacts', build_artifacts)
    correction_train.run(
        Namespace(
            store='weather',
            dem='dem',
            artifacts='artifacts',
            stations=str(path),
            source='era5',
            only=only,
        )
    )
    assert len(captured) == 1
    pd.testing.assert_frame_equal(captured[0][0], observations.drop(columns='unused'))
    assert captured[0][1] == only


def test_training_rejects_calendar_columns_without_date(tmp_path: Path):
    path = tmp_path / 'stations.parquet'
    pd.DataFrame(
        {
            'station_id': ['20046'],
            'lat': [80.6],
            'lon': [58.0],
            'year': [1980],
            'month': [1],
            'day': [1],
            'temperature': [-24.5],
        }
    ).to_parquet(path, index=False)
    with pytest.raises(pa.ArrowInvalid, match='date'):
        read_stations(
            get_file_store(path), columns=['station_id', 'lat', 'lon', 'date', 'temperature']
        )
