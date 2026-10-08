from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from weatheasy.storage import get_store, prepare_dataset, set_group_attrs, write_batch


def configured_api(sources: dict):
    pytest.importorskip('fastapi')
    from weatheasy.web import create_app
    from weatheasy.web.config import Settings, SourcesSettings

    return create_app(Settings(sources=SourcesSettings.model_validate(sources), _env_file=None))


def request(app: Any, path: str, **kwargs):
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        return client.get(path, **kwargs)


def test_era5_only_discovery_and_disabled_sources_need_no_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    from weatheasy.web import config

    def reject_store(*_args, **_kwargs):
        pytest.fail('Discovery and disabled sources must not open any store')

    monkeypatch.setattr(config, 'get_store', reject_store)
    app = configured_api({'era5': {'store': str(tmp_path / 'missing.zarr')}})
    variables = request(app, '/variables')
    assert variables.status_code == 200
    assert set(variables.json()) == {'era5'}
    assert variables.json()['era5']['TMP']['en'] == 'Temperature, °C'
    assert 'WIND' in variables.json()['era5']
    for source, variable in [('cfs2', 'TMP'), ('cmip6', 'tas')]:
        missing_query = request(app, f'/{source}')
        assert missing_query.status_code == 404
        response = request(
            app,
            f'/{source}',
            params={
                'lat': 55,
                'lon': 37,
                'begin': '2020-03-01',
                'end': '2020-03-01',
                'var': variable,
            },
        )
        assert response.status_code == 404
        assert response.json() == {'detail': 'Not Found'}


def test_http_sources_read_independent_stores(tmp_path: Path):
    sources = {}
    for source, group, variable, value, units in [
        ('era5', 'daily', 'TMP', 11, 'degC'),
        ('cfs2', 'reanalysis/flx', 'TMP', 22, 'degC'),
        ('cmip6', 'tas', 'tas', 280, 'K'),
    ]:
        path = tmp_path / f'{source}.zarr'
        store = get_store(path)
        dataset = xr.Dataset(
            {
                variable: (
                    ('time', 'latitude', 'longitude'),
                    np.full((1, 2, 2), value, np.float32),
                    {'units': units},
                )
            },
            coords={
                'time': [np.datetime64('2020-03-01')],
                'latitude': [55.0, 56.0],
                'longitude': [37.0, 38.0],
            },
        )
        write_batch(store, group, prepare_dataset(dataset, crs='EPSG:4326'))
        if source == 'cfs2':
            set_group_attrs(store, '', updated='2020-03-05')
        sources[source] = {'store': str(path)}

    app = configured_api(sources)
    for source, variable, expected in [
        ('era5', 'TMP', 11),
        ('cfs2', 'TMP', 22),
        ('cmip6', 'tas', 280),
    ]:
        response = request(
            app,
            f'/{source}',
            params={
                'lat': 55,
                'lon': 37,
                'begin': '2020-03-01',
                'end': '2020-03-01',
                'var': variable,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == [{variable: expected, 'date': '2020-03-01'}]


def test_source_environment_configuration_is_optional_and_lazy(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('fastapi')
    from weatheasy.web import config

    monkeypatch.setenv('WEATHEASY__SOURCES', '{}')
    monkeypatch.setenv('WEATHEASY__SOURCES__ERA5__STORE', 's3://weather/era5.zarr')
    monkeypatch.delenv('WEATHEASY__SOURCES__CFS2__STORE', raising=False)
    monkeypatch.delenv('WEATHEASY__SOURCES__CMIP6__STORE', raising=False)

    def reject_store(*_args, **_kwargs):
        pytest.fail('Parsing settings must not open stores or require S3 credentials')

    monkeypatch.setattr(config, 'get_store', reject_store)
    settings = config.Settings(_env_file=None)
    assert settings.sources.cfs2 is None
    assert settings.sources.cmip6 is None
    assert settings.sources.era5 is not None
    assert settings.sources.era5.store == 's3://weather/era5.zarr'


@pytest.mark.parametrize('enabled', [[], ['era5'], ['cfs2'], ['cmip6'], ['era5', 'cmip6']])
def test_only_configured_routes_and_schemas_are_registered(enabled: list[str]):
    app = configured_api({source: {'store': f'./{source}.zarr'} for source in enabled})
    # Schema generation must work before lifespan starts.
    schema = app.openapi()
    expected_paths = {'/variables', '/startupz', '/readyz', '/livez'}
    expected_paths.update(f'/{source}' for source in enabled)
    assert set(schema['paths']) == expected_paths
    for source, model in [
        ('era5', 'ERA5DataItem'),
        ('cfs2', 'CFS2DataItem'),
        ('cmip6', 'CMIP6DataItem'),
    ]:
        assert (model in schema['components']['schemas']) == (source in enabled)
    assert set(request(app, '/variables').json()) == set(enabled)


def test_applications_keep_routes_and_lifespan_state_independent():
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    era5_app = configured_api({'era5': {'store': './era5.zarr'}})
    cmip6_app = configured_api({'cmip6': {'store': './cmip6.zarr'}})
    with TestClient(era5_app) as era5, TestClient(cmip6_app) as cmip6:
        assert set(era5.get('/variables').json()) == {'era5'}
        assert set(cmip6.get('/variables').json()) == {'cmip6'}
        assert era5.get('/cmip6').status_code == 404
        assert cmip6.get('/era5').status_code == 404
        assert set(era5.get('/variables').json()) == {'era5'}


def test_variables_are_prepared_once_during_lifespan(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    from weatheasy.web import state

    validate_variables = state.Variables.model_validate
    calls = 0

    def track_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return validate_variables(*args, **kwargs)

    monkeypatch.setattr(state.Variables, 'model_validate', track_validation)
    app = configured_api({'era5': {'store': './era5.zarr'}})
    assert calls == 0
    with TestClient(app) as client:
        assert calls == 1
        for _ in range(2):
            response = client.get('/variables')
            assert response.status_code == 200
            assert response.json()['era5']['TMP']['en'] == 'Temperature, °C'
        assert calls == 1


def test_unknown_source_settings_are_rejected():
    pytest.importorskip('fastapi')
    from pydantic import ValidationError

    from weatheasy.web.config import SourcesSettings

    with pytest.raises(ValidationError, match='extra_forbidden'):
        SourcesSettings.model_validate({'er5': {'store': './era5.zarr'}})
