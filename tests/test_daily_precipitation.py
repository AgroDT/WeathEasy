import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from weatheasy.query import Coords, get_cfs2_data, get_cmip6_data, get_era5_data
from weatheasy.storage import open_dataset, prepare_dataset, write_batch
from weatheasy.util import get_storage


BEGIN, END = date(2020, 2, 28), date(2020, 3, 1)


def weather_store(tmp_path: Path, source: str):
    name = 'pr' if source == 'cmip6' else 'PRATE'
    root = get_storage(str(tmp_path / source))
    scale = 21600 if source == 'cfs2' else 86400
    raw = np.array([8, np.nan, 2], dtype=np.float32) / scale
    dataset = prepare_dataset(
        xr.Dataset(
            {
                name: (
                    ('time', 'latitude', 'longitude'),
                    np.broadcast_to(raw[:, None, None], (3, 2, 2)).copy(),
                    {'units': 'kg m-2 s-1'},
                )
            },
            coords={
                'time': np.arange('2020-02-28', '2020-03-02', dtype='datetime64[D]'),
                'latitude': [54.0, 55.0],
                'longitude': [36.0, 37.0],
            },
        ),
        crs='EPSG:4326',
    )
    if source == 'cfs2':
        write_batch(root.store, 'reanalysis/flx', dataset.isel(time=slice(0, 2)))
        write_batch(root.store, 'forecast/flx', dataset.isel(time=slice(2, None)))
        root[''].attrs['updated'] = '2020-03-04'  # forecast begins March 1
        path = 'reanalysis/flx'
    else:
        path = 'daily' if source == 'era5' else 'pr'
        write_batch(root.store, path, dataset)
    return root, name, path


@pytest.mark.parametrize('source', ['cfs2', 'era5', 'cmip6'])
def test_daily_getter_and_csv_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str):
    from weatheasy.__main__ import main

    root, name, _ = weather_store(tmp_path, source)
    getter = {'cfs2': get_cfs2_data, 'era5': get_era5_data, 'cmip6': get_cmip6_data}[source]
    result = getter(store=root.store, begin=BEGIN, end=END, coords=Coords(55, 37), variables=[name])
    np.testing.assert_allclose(result, [[8, np.nan, 2]])
    target = tmp_path / 'result.csv'
    argv = [
        'weatheasy',
        'query',
        source,
        '-d',
        str(tmp_path / source),
        '-o',
        str(target),
        BEGIN.isoformat(),
        END.isoformat(),
        '55',
        '37',
        name,
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    main(configure_logging=False)
    expected = f'DATE,{name}\n2020-02-28,8\n2020-02-29,NA\n2020-03-01,2\n'
    assert target.read_text() == expected

    stdout = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    argv[6] = 'stdout'
    main(configure_logging=False)
    assert not stdout.closed
    assert stdout.getvalue() == expected


@pytest.mark.parametrize('source', ['cmip6', 'cfs2'])
def test_http_precipitation_units_and_response_shape(
    tmp_path: Path,
    source: str,
):
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    from weatheasy.web import create_app
    from weatheasy.web.config import Settings, SourcesSettings

    root, name, _ = weather_store(tmp_path, source)
    settings = Settings(
        sources=SourcesSettings.model_validate({source: {'store': str(root.store.root)}}),
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(
            f'/{source}',
            params={
                'begin': BEGIN.isoformat(),
                'end': END.isoformat(),
                'lat': 55,
                'lon': 37,
                'var': name,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == [
            {name: 8, 'date': '2020-02-28'},
            {name: None, 'date': '2020-02-29'},
            {name: 2, 'date': '2020-03-01'},
        ]
        variables = client.get('/variables').json()
        assert 'mm/day' in variables[source][name]['en']


@pytest.mark.parametrize('source', ['cfs2', 'era5'])
def test_raw_and_corrected_precipitation_have_same_units(tmp_path: Path, source: str):
    root, _, path = weather_store(tmp_path, source)
    with open_dataset(root.store, path) as raw:
        corrected = raw.rename({'PRATE': 'PRATE_corr'}).load()
    corrected['PRATE_corr'][:] = 12
    corrected.PRATE_corr.attrs['units'] = 'mm/day'
    correction_path = 'reanalysis/correction' if source == 'cfs2' else 'correction'
    write_batch(root.store, correction_path, corrected)
    getter = get_cfs2_data if source == 'cfs2' else get_era5_data
    values = getter(
        store=root.store,
        begin=BEGIN,
        end=BEGIN,
        coords=Coords(55, 37),
        variables=['PRATE_corr', 'PRATE'],
    )
    np.testing.assert_allclose(values, [[12], [8]])
