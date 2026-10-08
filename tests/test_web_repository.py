from datetime import date
from threading import Barrier, Event, get_ident

import numpy as np
import pytest


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('source', ['era5', 'cfs2', 'cmip6'])
async def test_store_created_once_per_lifespan_off_loop(
    source: str, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip('fastapi')
    from anyio import Event as AsyncEvent, create_task_group, fail_after, from_thread
    from zarr.storage import MemoryStore

    from weatheasy.web import create_app, state
    from weatheasy.web.config import Settings, SourcesSettings

    loop_thread = get_ident()
    started = AsyncEvent()
    release = Event()
    stores_created = []
    uri = f's3://weather/{source}.zarr'

    def create_store(location: str):
        assert get_ident() != loop_thread, 'Store initialization must run off the event loop'
        assert location == uri
        store = MemoryStore()
        stores_created.append(store)
        from_thread.run_sync(started.set)
        assert release.wait(timeout=5), 'Event loop must run while store initialization is blocked'
        return store

    monkeypatch.setattr(state, 'get_store', create_store)
    settings = Settings(
        sources=SourcesSettings.model_validate({source: {'store': uri}}), _env_file=None
    )
    app = create_app(settings)
    assert stores_created == []
    startup_finished = AsyncEvent()

    async def run_lifespan():
        async with app.router.lifespan_context(app) as app_state:
            assert app_state is not None
            assert len(stores_created) == 1
            assert set(app_state) == {'settings', 'variables', f'{source}_repository'}
            startup_finished.set()

    try:
        async with create_task_group() as tasks:
            tasks.start_soon(run_lifespan)
            with fail_after(3):
                await started.wait()
            assert not startup_finished.is_set()
            assert len(stores_created) == 1
            release.set()
    finally:
        release.set()

    assert len(stores_created) == 1
    # Reusing settings for another lifespan must create a fresh store.
    async with app.router.lifespan_context(app):
        assert len(stores_created) == 2
        assert stores_created[0] is not stores_created[1]


@pytest.mark.anyio
async def test_store_initialization_failure_aborts_startup(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('fastapi')
    from weatheasy.web import create_app, state
    from weatheasy.web.config import Settings, SourceSettings, SourcesSettings

    def fail_store(_uri: str):
        msg = 'Store initialization failed'
        raise OSError(msg)

    monkeypatch.setattr(state, 'get_store', fail_store)
    app = create_app(
        Settings(sources=SourcesSettings(era5=SourceSettings(store='s3://weather/era5.zarr')))
    )
    with pytest.raises(OSError, match='Store initialization failed'):
        async with app.router.lifespan_context(app):
            pytest.fail('Startup must not complete without initialized stores')


@pytest.mark.anyio
async def test_repository_reads_ready_store_concurrently():
    pytest.importorskip('fastapi')
    from anyio import CapacityLimiter, create_task_group
    from zarr.storage import MemoryStore

    from weatheasy.query import Coords
    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import WeatherRepository

    loop_thread = get_ident()
    store = MemoryStore()
    readers = Barrier(4)
    query = DataQuery(
        coords=Coords(latitude=55, longitude=37),
        begin=date(2020, 3, 1),
        end=date(2020, 3, 2),
        variables=['TMP'],
    )

    def getter(**kwargs):
        assert kwargs == {**query, 'store': store}
        assert get_ident() != loop_thread
        readers.wait(timeout=5)
        return np.array([[11, 12]], dtype=np.float32)

    limiter = CapacityLimiter(4)
    repository = WeatherRepository(store, getter, limiter)
    results = []

    async def read():
        results.append(await repository.read(query))

    for _ in range(2):
        async with create_task_group() as tasks:
            for _ in range(4):
                tasks.start_soon(read)
    assert len(results) == 8
    for result in results:
        np.testing.assert_array_equal(result, [[11, 12]])
    assert limiter.borrowed_tokens == 0
