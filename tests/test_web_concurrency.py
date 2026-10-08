from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest


@pytest.mark.parametrize(
    ('source', 'variable'), [('era5', 'TMP'), ('cfs2', 'TMP'), ('cmip6', 'tas')]
)
def test_storage_reads_leave_default_pool_available(
    source: str, variable: str, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    from anyio import to_thread
    from fastapi.responses import StreamingResponse
    from starlette.testclient import TestClient

    from weatheasy.web import create_app, state
    from weatheasy.web.config import Settings, SourcesSettings

    started = Event()
    release = Event()

    def slow_getter(**_kwargs):
        started.set()
        if not release.wait(timeout=10):
            pytest.fail('Storage read was not released')
        return np.array([[11, 12]], dtype=np.float32)

    monkeypatch.setattr(state, f'get_{source}_data', slow_getter)
    settings = Settings(
        sources=SourcesSettings.model_validate({source: {'store': './unused.zarr'}}),
        _env_file=None,
    )
    # Avoid opening storage; exercise the real route, dependencies and worker dispatch.
    getattr(settings.sources, source).__dict__['zarr_store'] = object()
    app = create_app(settings)

    @app.get('/prepared')
    async def prepared() -> StreamingResponse:
        return StreamingResponse(iter(['[', '11', ']']), media_type='application/json')

    async def configure_default_pool():
        limiter = to_thread.current_default_thread_limiter()
        limiter.total_tokens = 1
        return limiter

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as executor:
        assert client.portal is not None
        default_limiter = client.portal.call(configure_default_pool)
        read_limiter = client.app_state[f'{source}_repository'].read_limiter
        read_limiter.total_tokens = 1
        pending_read = executor.submit(
            client.get,
            f'/{source}',
            params={
                'lat': 55,
                'lon': 37,
                'begin': '2020-03-01',
                'end': '2020-03-02',
                'var': variable,
            },
        )
        try:
            assert started.wait(timeout=3), 'The storage worker must actually be blocked'
            assert read_limiter.borrowed_tokens == 1
            assert default_limiter.borrowed_tokens == 0
            discovery = executor.submit(client.get, '/variables').result(timeout=3)
            streamed = executor.submit(client.get, '/prepared').result(timeout=3)
            assert discovery.status_code == 200
            assert variable in discovery.json()[source]
            assert streamed.status_code == 200
            assert streamed.json() == [11]
            assert not pending_read.done()
        finally:
            release.set()
        response = pending_read.result(timeout=3)
        assert response.status_code == 200
        assert response.json() == [
            {variable: 11, 'date': '2020-03-01'},
            {variable: 12, 'date': '2020-03-02'},
        ]
