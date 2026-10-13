from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from weatheasy._cli.commands import correction_train
from weatheasy.file_storage import get_file_store


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
    observations[['elevation', 'slope_we', 'slope_sn']] = [123.0, 0.1, 0.2]
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
        _artifacts: object,
        *,
        only: str | None,
    ):
        captured.append((stations, only))

    monkeypatch.setattr(correction_train.training, 'build_artifacts', build_artifacts)
    correction_train.run(
        Namespace(
            store='weather',
            artifacts='artifacts',
            stations=str(path),
            source='era5',
            only=only,
        )
    )
    assert len(captured) == 1
    pd.testing.assert_frame_equal(
        captured[0][0], observations.drop(columns='unused'), check_like=True
    )
    assert captured[0][1] == only
