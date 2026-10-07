from pathlib import Path

import pytest


@pytest.mark.parametrize('path', ['/startupz', '/readyz', '/livez'])
def test_health_probes_do_not_access_storage(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    from weatheasy.web import config, create_app

    settings = config.Settings(
        sources=config.SourcesSettings(
            era5=config.SourceSettings(store=str(tmp_path / 'missing-store'))
        ),
        _env_file=None,
    )

    def fail_storage_access(*_args, **_kwargs):
        pytest.fail('Health probes must not access configuration or weather storage')

    monkeypatch.setattr(config, 'get_store', fail_storage_access)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert response.headers['content-type'] == 'application/json'
