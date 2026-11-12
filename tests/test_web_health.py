from pathlib import Path

import pytest


@pytest.mark.parametrize('path', ['/startupz', '/readyz', '/livez'])
def test_health_probes_do_not_access_storage(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient
    from zarr.storage import MemoryStore

    from weatheasy.web import config, create_app, repository

    settings = config.Settings(
        sources=config.SourcesSettings(
            era5=config.SourceSettings(store=str(tmp_path / 'missing-store'))
        ),
        _env_file=None,
    )

    def fail_storage_access(*_args, **_kwargs):
        pytest.fail('Health probes must not access configuration or weather storage')

    monkeypatch.setattr(repository, 'get_store', lambda _uri, **_kwargs: MemoryStore())
    app = create_app(settings)
    with TestClient(app) as client:
        monkeypatch.setattr(repository, 'get_store', fail_storage_access)
        response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert response.headers['content-type'] == 'application/json'


@pytest.mark.parametrize('prefix', ['', '/api'])
def test_proxy_prefix_is_used_in_docs_but_not_health_paths(prefix: str):
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    from weatheasy.web import config, create_app

    settings = config.Settings(root_path=prefix, sources=config.SourcesSettings(), _env_file=None)
    with TestClient(create_app(settings)) as client:
        docs = client.get('/docs')
        assert docs.status_code == 200
        assert f"url: '{prefix}/openapi.json'" in docs.text
        schema = client.get('/openapi.json')
        assert schema.status_code == 200
        assert schema.json().get('servers', []) == ([{'url': '/api'}] if prefix else [])
        assert '/variables' in schema.json()['paths']
        for path in ['/startupz', '/readyz', '/livez']:
            response = client.get(path)
            assert response.status_code == 200
            assert response.json() == {'status': 'ok'}
