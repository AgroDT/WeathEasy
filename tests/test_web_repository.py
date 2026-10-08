from __future__ import annotations

from datetime import date
from threading import Event, get_ident
from typing import TYPE_CHECKING

import numpy as np
import pytest


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import ReadPlan


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def query() -> DataQuery:
    pytest.importorskip('fastapi')
    from weatheasy.query import Coords
    from weatheasy.web.models import DataQuery

    return DataQuery(
        coords=Coords(latitude=55, longitude=37),
        begin=date(2020, 3, 1),
        end=date(2020, 3, 2),
        variables=['TMP'],
    )


def constant_plan(values: NDArray[np.float32]) -> ReadPlan:
    from weatheasy.web.repository import ReadPlan

    class ConstantPlan(ReadPlan):
        async def read(self, _first: date, _last: date, /):
            return values

    return ConstantPlan([], [])


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
    from anyio import CapacityLimiter, Event as AsyncEvent, create_task_group
    from zarr.storage import MemoryStore

    from weatheasy.query import Coords
    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import WeatherRepository

    loop_thread = get_ident()
    store = MemoryStore()
    all_started = AsyncEvent()
    active = 0
    query = DataQuery(
        coords=Coords(latitude=55, longitude=37),
        begin=date(2020, 3, 1),
        end=date(2020, 3, 2),
        variables=['TMP'],
    )

    class ConcurrentRepository(WeatherRepository):
        async def _prepare(self, data_query: DataQuery):
            nonlocal active
            assert data_query == query
            assert self._reader.store is store
            assert get_ident() == loop_thread
            active += 1
            if active == 4:
                all_started.set()
            await all_started.wait()
            return constant_plan(np.array([[11, 12]], dtype=np.float32))

    limiter = CapacityLimiter(4)
    repository = ConcurrentRepository(store, limiter)
    results = []

    async def read():
        results.append(await repository.read(query))

    for _ in range(2):
        active = 0
        all_started = AsyncEvent()
        async with create_task_group() as tasks:
            for _ in range(4):
                tasks.start_soon(read)
    assert len(results) == 8
    for result in results:
        np.testing.assert_array_equal(result, [[11, 12]])
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
@pytest.mark.parametrize(('read_timeout', 'operation_timeout'), [(2, 0.05), (0.05, 2)])
async def test_hung_storage_operation_times_out_and_releases_read_slot(
    query: DataQuery, read_timeout: float, operation_timeout: float
):
    pytest.importorskip('fastapi')
    import asyncio

    from anyio import CapacityLimiter, fail_after
    from zarr.storage import MemoryStore

    from weatheasy.web.repository import Era5Repository, StorageReadTimeoutError, WeatherRepository

    cancelled = Event()

    class HungStore(MemoryStore):
        async def get(self, *_args, **_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    limiter = CapacityLimiter(1)
    repository = Era5Repository(
        HungStore(),
        limiter,
        read_timeout=read_timeout,
        operation_timeout=operation_timeout,
    )
    with fail_after(1), pytest.raises(StorageReadTimeoutError):
        await repository.read(query)
    assert cancelled.is_set(), 'The underlying asynchronous operation must be cancelled'
    assert limiter.borrowed_tokens == 0

    class ReadyRepository(WeatherRepository):
        async def _prepare(self, _query: DataQuery):
            return constant_plan(np.array([[42]], dtype=np.float32))

    ready = ReadyRepository(MemoryStore(), limiter)
    with fail_after(3):
        np.testing.assert_array_equal(await ready.read(query), [[42]])


@pytest.mark.anyio
async def test_cancelled_running_read_cancels_io_and_releases_slot(query: DataQuery):
    pytest.importorskip('fastapi')
    from anyio import (
        CancelScope,
        CapacityLimiter,
        Event as AsyncEvent,
        create_task_group,
        fail_after,
    )
    from zarr.storage import MemoryStore

    from weatheasy.web.repository import Era5Repository

    started = AsyncEvent()
    cancelled = AsyncEvent()
    finished = AsyncEvent()
    scope = CancelScope()
    limiter = CapacityLimiter(1)
    returned = []

    class HungStore(MemoryStore):
        async def get(self, *_args, **_kwargs):
            started.set()
            try:
                await AsyncEvent().wait()
            finally:
                cancelled.set()

    repository = Era5Repository(HungStore(), limiter)

    async def read():
        with scope:
            returned.append(await repository.read(query))
        finished.set()

    with fail_after(1):
        async with create_task_group() as tasks:
            tasks.start_soon(read)
            await started.wait()
            assert limiter.borrowed_tokens == 1
            scope.cancel()
            await finished.wait()
    assert cancelled.is_set()
    assert returned == []
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
async def test_queued_read_times_out_without_starting_io(query: DataQuery):
    pytest.importorskip('fastapi')
    from anyio import CapacityLimiter, fail_after
    from zarr.storage import MemoryStore

    from weatheasy.web.repository import StorageReadTimeoutError, WeatherRepository

    class FailingRepository(WeatherRepository):
        async def _prepare(self, _query: DataQuery):
            pytest.fail('An expired queued request must never start I/O')

    limiter = CapacityLimiter(1)
    repository = FailingRepository(MemoryStore(), limiter, read_timeout=0.05)
    borrower = object()
    await limiter.acquire_on_behalf_of(borrower)
    try:
        with fail_after(3), pytest.raises(StorageReadTimeoutError):
            await repository.read(query)
        assert limiter.borrowed_tokens == 1
    finally:
        limiter.release_on_behalf_of(borrower)
    assert limiter.borrowed_tokens == 0


def test_storage_timeout_returns_504_before_streaming(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('fastapi')
    import asyncio

    from starlette.testclient import TestClient
    from zarr.storage import MemoryStore

    from weatheasy.web import create_app, state
    from weatheasy.web.config import Settings, SourceSettings, SourcesSettings

    cancelled = Event()

    class HungStore(MemoryStore):
        async def get(self, *_args, **_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    monkeypatch.setattr(state, 'get_store', lambda _uri: HungStore())
    app = create_app(
        Settings(
            sources=SourcesSettings(era5=SourceSettings(store='s3://weather/era5.zarr')),
            storage_read_timeout=2,
            storage_operation_timeout=0.05,
            _env_file=None,
        )
    )
    with TestClient(app) as client:
        response = client.get(
            '/era5',
            params={'lat': 55, 'lon': 37, 'begin': '2020-03-01', 'end': '2020-03-02', 'var': 'TMP'},
        )
        assert response.status_code == 504
        assert response.json() == {'detail': 'Storage read timed out'}
        assert cancelled.is_set()
        assert client.app_state['era5_repository'].read_limiter.borrowed_tokens == 0
        assert client.get('/variables').status_code == 200
