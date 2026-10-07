import asyncio
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient


@pytest.mark.parametrize('path', ['/startupz', '/readyz', '/livez'])
def test_health_probes_do_not_access_storage(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    monkeypatch.setenv('WEATHEASY__DATA_ROOT', str(tmp_path / 'missing-store'))
    from weatheasy.web import app, config, controller

    def fail_storage_access(*_args, **_kwargs):
        pytest.fail('Health probes must not access configuration or weather storage')

    monkeypatch.setattr(config, 'get_store', fail_storage_access)
    monkeypatch.setattr(controller, 'get_config', fail_storage_access)
    monkeypatch.setattr(controller, 'get_variables', fail_storage_access)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            return await client.get(path)

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert response.headers['content-type'] == 'application/json'
