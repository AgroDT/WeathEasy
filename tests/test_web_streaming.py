from __future__ import annotations

import json
from copy import copy
from datetime import date, timedelta
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from weatheasy.query import Coords
from weatheasy.storage import prepare_dataset, set_group_attrs, write_batch


if TYPE_CHECKING:
    from anyio import CapacityLimiter
    from starlette.types import Message

    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import WeatherRepository


@pytest.fixture(name='anyio_backend')
def get_anyio_backend():
    return 'asyncio'


@pytest.fixture(name='prepared_stream', params=['era5', 'cfs2', 'cmip6'])
def create_prepared_stream(request: pytest.FixtureRequest):
    pytest.importorskip('fastapi')
    from weatheasy.web.models import DataQuery

    source = request.param
    variable = 'tas' if source == 'cmip6' else 'TMP'
    dates = np.arange('2020-01-01', '2020-05-10', dtype='datetime64[D]')
    values = np.arange(130, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    # A missing date inside the second batch must still appear as null.
    keep = np.arange(130) != 65
    dataset = prepare_dataset(
        xr.Dataset(
            {variable: (('time', 'latitude', 'longitude'), values[keep], {'units': 'degC'})},
            coords={'time': dates[keep], 'latitude': [54.5, 55.5], 'longitude': [36.5, 37.5]},
        ),
        crs='EPSG:4326',
    )
    store = RecordingStore()
    if source == 'cfs2':
        write_batch(store, 'reanalysis/flx', dataset.isel(time=slice(0, 69)))
        write_batch(store, 'forecast/flx', dataset.isel(time=slice(69, None)))
        set_group_attrs(store, '', updated='2020-03-14')  # Forecast begins March 11 (day 70).
    else:
        write_batch(store, 'daily' if source == 'era5' else variable, dataset)
    query = DataQuery(
        coords=Coords(55, 37), begin=date(2020, 1, 1), end=date(2020, 5, 9), variables=[variable]
    )
    store.read_keys.clear()
    return source, store, query


class RecordingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.read_keys = []

    @override
    def with_read_only(self, read_only: bool = False) -> RecordingStore:
        store = copy(self)
        store._read_only = read_only
        return store

    async def get(self, key: str, *args, **kwargs):
        self.read_keys.append(key)
        return await super().get(key, *args, **kwargs)


def repository_for(
    source: str, store: MemoryStore, limiter: CapacityLimiter, **kwargs
) -> WeatherRepository:
    from weatheasy.web.repository import Cfs2Repository, Cmip6Repository, Era5Repository

    return {'era5': Era5Repository, 'cfs2': Cfs2Repository, 'cmip6': Cmip6Repository}[source](
        store, limiter, **kwargs
    )


@pytest.mark.anyio
async def test_http_stream_sends_first_batch_before_reading_rest_and_loads_metadata_once(
    prepared_stream: tuple[str, RecordingStore, DataQuery],
):
    from anyio import CapacityLimiter

    from weatheasy.web.config import Settings
    from weatheasy.web.controller import data_response

    source, store, query = prepared_stream
    limiter = CapacityLimiter(1)
    repository = repository_for(source, store, limiter)
    response = await data_response(repository, query, Settings(_env_file=None))
    variable = query['variables'][0]
    messages = []
    metadata = [key for key in store.read_keys if key.endswith('/zarr.json')]

    async def send(message: Message):
        assert limiter.borrowed_tokens == 0, 'A slow sender must not hold a storage slot'
        if message['type'] == 'http.response.body' and not any(
            item['type'] == 'http.response.body' for item in messages
        ):
            chunks = [key for key in store.read_keys if f'/{variable}/c/' in key]
            assert len(chunks) == 64, 'First output must precede storage reads for later days'
            await limiter.acquire()
            limiter.release()
        messages.append(message)

    await response.stream_response(send)
    assert messages[0]['status'] == 200
    body = b''.join(item['body'] for item in messages if item['type'] == 'http.response.body')
    expected = [
        {
            variable: None if index == 65 else index,
            'date': str(query['begin'] + timedelta(days=index)),
        }
        for index in range(130)
    ]
    assert json.loads(body) == expected
    assert [key for key in store.read_keys if key.endswith('/zarr.json')] == metadata
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
async def test_late_storage_timeout_aborts_json_after_first_batch(
    prepared_stream: tuple[str, RecordingStore, DataQuery],
    monkeypatch: pytest.MonkeyPatch,
):
    from anyio import CapacityLimiter, Event, fail_after

    from weatheasy.web.config import Settings
    from weatheasy.web.controller import data_response
    from weatheasy.web.repository import StorageReadTimeoutError

    source, store, query = prepared_stream
    variable = query['variables'][0]
    limiter = CapacityLimiter(1)
    original_get = store.get
    first_sent = False
    cancelled = Event()

    async def blocked_get(key: str, *args, **kwargs):
        if first_sent and f'/{variable}/c/' in key:
            try:
                await Event().wait()
            finally:
                cancelled.set()
        return await original_get(key, *args, **kwargs)

    monkeypatch.setattr(store, 'get', blocked_get)
    repository = repository_for(source, store, limiter, operation_timeout=1)
    response = await data_response(repository, query, Settings(_env_file=None))
    # Trigger the short deadline only after the first batch has succeeded.
    repository._reader.operation_timeout = 0.02
    messages = []

    async def send(message: Message):
        nonlocal first_sent
        messages.append(message)
        if message['type'] == 'http.response.body':
            first_sent = True

    with fail_after(1), pytest.raises(StorageReadTimeoutError):
        await response.stream_response(send)
    assert messages[0]['status'] == 200
    body = b''.join(item['body'] for item in messages if item['type'] == 'http.response.body')
    assert len(json.loads(body + b']')) == 64
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)
    assert cancelled.is_set()
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
async def test_cancel_after_first_batch_cancels_next_chunk_reads(
    prepared_stream: tuple[str, RecordingStore, DataQuery],
    monkeypatch: pytest.MonkeyPatch,
):
    from anyio import CancelScope, CapacityLimiter, Event, create_task_group, fail_after

    from weatheasy.web.config import Settings
    from weatheasy.web.controller import data_response

    source, store, query = prepared_stream
    variable = query['variables'][0]
    limiter = CapacityLimiter(1)
    original_get = store.get
    started, stopped, finished = Event(), Event(), Event()
    first_sent = False
    scope = CancelScope()

    async def blocked_get(key: str, *args, **kwargs):
        if first_sent and f'/{variable}/c/' in key:
            started.set()
            try:
                await Event().wait()
            finally:
                stopped.set()
        return await original_get(key, *args, **kwargs)

    monkeypatch.setattr(store, 'get', blocked_get)
    response = await data_response(
        repository_for(source, store, limiter), query, Settings(_env_file=None)
    )

    async def send(message: Message):
        nonlocal first_sent
        if message['type'] == 'http.response.body':
            assert limiter.borrowed_tokens == 0
            first_sent = True

    async def run_response():
        with scope:
            await response.stream_response(send)
            pytest.fail('Cancelled streaming must not finish normally')
        finished.set()

    with fail_after(1):
        async with create_task_group() as tasks:
            tasks.start_soon(run_response)
            await started.wait()
            assert limiter.borrowed_tokens == 1
            scope.cancel()
            await finished.wait()
    assert stopped.is_set()
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
async def test_storage_budget_accumulates_read_time_and_excludes_sender_time(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip('fastapi')
    from anyio import CapacityLimiter

    from weatheasy.web import repository as module
    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import ReadPlan, StorageReadTimeoutError, WeatherRepository

    elapsed = 0.0
    reads = 0
    preparations = 0
    monkeypatch.setattr(module, 'monotonic', lambda: elapsed)

    class MeasuredPlan(ReadPlan):
        async def read(self, _first: date, _last: date, /):
            nonlocal elapsed, reads
            reads += 1
            elapsed += 0.04
            return np.array([[reads]], dtype=np.float32)

    class MeasuredRepository(WeatherRepository):
        async def _prepare(self, _query: DataQuery):
            nonlocal preparations
            preparations += 1
            return MeasuredPlan([], [])

    limiter = CapacityLimiter(1)
    repository = MeasuredRepository(MemoryStore(), limiter, read_timeout=0.1)
    query = DataQuery(
        coords=Coords(55, 37), begin=date(2020, 1, 1), end=date(2020, 1, 4), variables=['TMP']
    )
    batches = repository.stream(query, batch_days=1)
    np.testing.assert_array_equal(await anext(batches), [[1]])
    assert limiter.borrowed_tokens == 0
    elapsed += 100  # Time spent by the consumer outside storage operations.
    np.testing.assert_array_equal(await anext(batches), [[2]])
    assert limiter.borrowed_tokens == 0
    with pytest.raises(StorageReadTimeoutError):
        await anext(batches)
    assert preparations == 1
    assert reads == 3, 'A fresh budget for each batch would let the third batch through'
    assert limiter.borrowed_tokens == 0
