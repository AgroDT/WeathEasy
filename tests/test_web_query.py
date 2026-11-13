from __future__ import annotations

from copy import copy
from datetime import date
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

from weatheasy import query as synchronous
from weatheasy.query import Coords
from weatheasy.storage import (
    StorageError,
    open_dataset,
    prepare_dataset,
    set_group_attrs,
    write_batch,
)


if TYPE_CHECKING:
    from anyio import CapacityLimiter

    from weatheasy.web.models import DataQuery
    from weatheasy.web.repository import WeatherRepository


@pytest.fixture(name='anyio_backend')
def get_anyio_backend():
    return 'asyncio'


@pytest.fixture(name='prepared_source', params=['era5', 'cfs2', 'cmip6'])
def create_prepared_source(request: pytest.FixtureRequest):
    return prepared_store(request.param)


def prepared_store(source: str) -> tuple[str, RecordingStore, list[str]]:
    store = RecordingStore()
    # Missing Feb 29; distinguish each time, latitude and longitude.
    values = (
        np.arange(3, dtype=np.float32)[:, None, None] * 100
        + np.arange(2, dtype=np.float32)[None, :, None] * 10
        + np.arange(4, dtype=np.float32)[None, None, :]
    )
    scale = 21600 if source == 'cfs2' else 86400
    temperature, precipitation = ('tas', 'pr') if source == 'cmip6' else ('TMP', 'PRATE')
    fields = {
        temperature: (('time', 'latitude', 'longitude'), values, {'units': 'degC'}),
        precipitation: (('time', 'latitude', 'longitude'), values / scale, {'units': 'kg m-2 s-1'}),
    }
    if source != 'cmip6':
        fields['PRATE_corr'] = (
            ('time', 'latitude', 'longitude'),
            values + 500,
            {'units': 'mm/day'},
        )
    dataset = prepare_dataset(
        xr.Dataset(
            fields,
            coords={
                'time': np.array(['2020-02-28', '2020-03-01', '2020-03-02'], dtype='datetime64[D]'),
                'latitude': [54.5, 55.5],
                'longitude': [170, 175, -180, -175],
            },
        ),
        crs='EPSG:4326',
    )
    if source == 'cfs2':
        write_batch(
            store, 'reanalysis/flx', dataset[[temperature, precipitation]].isel(time=slice(0, 1))
        )
        write_batch(
            store, 'forecast/flx', dataset[[temperature, precipitation]].isel(time=slice(1, None))
        )
        write_batch(store, 'reanalysis/correction', dataset[['PRATE_corr']].isel(time=slice(0, 1)))
        set_group_attrs(store, '', updated='2020-03-04')  # Forecast starts March 1.
        zarr.open_group(store, path='reanalysis/pgb', mode='a')
        humidity = dataset[[temperature]].rename({temperature: 'RH'})
        humidity['RH'] += 1000
        humidity.RH.attrs['units'] = '%'
        write_batch(store, 'forecast/pgb', humidity.isel(time=slice(1, None)))
    elif source == 'era5':
        write_batch(store, 'daily', dataset[[temperature, precipitation]])
        write_batch(store, 'correction', dataset[['PRATE_corr']])
    else:
        for variable in fields:
            write_batch(store, variable, dataset[[variable]])
    variables = [precipitation, temperature]
    if source == 'cfs2':
        variables.append('RH')
    if source != 'cmip6':
        variables.insert(0, 'PRATE_corr')
    return source, store, variables


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


def make_query(variables: list[str]) -> DataQuery:
    pytest.importorskip('fastapi')
    from weatheasy.web.models import DataQuery

    return DataQuery(
        coords=Coords(55, -177.5),  # Both nearest ties choose the north/east cell.
        begin=date(2020, 2, 28),
        end=date(2020, 3, 2),
        variables=variables,
    )


def make_repository(
    source: str, store: MemoryStore, limiter: CapacityLimiter | None = None
) -> WeatherRepository:
    from anyio import CapacityLimiter

    from weatheasy.web.repository import Cfs2Repository, Cmip6Repository, Era5Repository

    repositories = {'cfs2': Cfs2Repository, 'cmip6': Cmip6Repository, 'era5': Era5Repository}
    return repositories[source](store, limiter if limiter is not None else CapacityLimiter(40))


@pytest.mark.anyio
async def test_async_sources_preserve_values_dates_order_and_corrections(
    prepared_source: tuple[str, RecordingStore, list[str]],
):
    pytest.importorskip('fastapi')
    from anyio import to_thread

    source, store, variables = prepared_source
    query = make_query(variables)
    actual = await make_repository(source, store).read(query)
    # Independent oracle: north/east cell is row 1, column 3.
    expected = [[13, np.nan, 113, 213], [13, np.nan, 113, 213]]
    if source != 'cmip6':
        expected.insert(
            0, [513, np.nan, np.nan, np.nan] if source == 'cfs2' else [513, np.nan, 613, 713]
        )
    if source == 'cfs2':
        expected.append([np.nan, np.nan, 1113, 1213])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    sync_getter = getattr(synchronous, f'get_{source}_data')
    reference = await to_thread.run_sync(lambda: sync_getter(store=store, **query))
    np.testing.assert_allclose(actual, reference, rtol=1e-6)


@pytest.mark.anyio
async def test_async_sources_preserve_coordinate_errors(
    prepared_source: tuple[str, RecordingStore, list[str]],
):
    pytest.importorskip('fastapi')
    from weatheasy.error import CoordsError

    source, store, variables = prepared_source
    query = make_query(variables)
    query['coords'] = Coords(55, 0)
    with pytest.raises(CoordsError, match='out of'):
        await make_repository(source, store).read(query)


@pytest.mark.anyio
@pytest.mark.parametrize('defect', ['incomplete', 'length', 'schema', 'field_length'])
async def test_async_sources_reject_uncommitted_or_invalid_groups(
    prepared_source: tuple[str, RecordingStore, list[str]], defect: str
):
    pytest.importorskip('fastapi')

    source, store, variables = prepared_source
    path = {'era5': 'daily', 'cfs2': 'reanalysis/flx', 'cmip6': 'pr'}[source]
    group = zarr.open_group(store, path=path, mode='a')
    if defect == 'incomplete':
        group.attrs['write_state'] = 'writing'
    elif defect == 'length':
        group.attrs['committed_length'] = 99
    elif defect == 'schema':
        group.attrs['storage_schema'] = 'unknown'
    else:
        name = 'pr' if source == 'cmip6' else 'PRATE'
        array = group.get_array(name)
        array.resize((array.shape[0] + 1, *array.shape[1:]))
    with pytest.raises(StorageError):
        await make_repository(source, store).read(make_query(variables))


@pytest.mark.anyio
async def test_async_point_reads_only_requested_weather_chunks(
    prepared_source: tuple[str, RecordingStore, list[str]],
):
    pytest.importorskip('fastapi')

    source, store, _ = prepared_source
    variable = 'tas' if source == 'cmip6' else 'TMP'
    group = {'era5': 'daily', 'cfs2': 'forecast/flx', 'cmip6': 'tas'}[source]
    # Explicit weather layout contract: archives128 days, CFS2 forecast64 days.
    chunk_size = 64 if source == 'cfs2' else 128
    initial_days = 2 if source == 'cfs2' else 3
    dates = np.arange('2020-03-03', '2020-07-11', dtype='datetime64[D]')
    with open_dataset(store, group) as existing:
        assert existing.sizes['time'] == initial_days
        template = existing.isel(time=slice(0, 1)).load()
    appended = xr.concat([template] * 130, dim='time').assign_coords(time=dates)
    values = (
        np.arange(130, dtype=np.float32)[:, None, None] * 100
        + np.arange(2, dtype=np.float32)[None, :, None] * 10
        + np.arange(4, dtype=np.float32)[None, None, :]
    )
    appended[variable].data = values
    write_batch(store, group, prepare_dataset(appended, crs='EPSG:4326'))
    query = make_query([variable])
    requested_index = chunk_size - initial_days
    query['begin'] = query['end'] = date.fromisoformat(str(dates[requested_index]))
    store.read_keys.clear()
    actual = await make_repository(source, store).read(query)
    np.testing.assert_array_equal(actual, [[requested_index * 100 + 13]])
    prefix = f'{group}/{variable}'
    weather_names = ('TMP', 'PRATE', 'PRATE_corr', 'RH', 'tas', 'pr')
    chunks = [key for key in store.read_keys if any(f'/{name}/c/' in key for name in weather_names)]
    assert chunks == [f'{prefix}/c/1/0/0']


@pytest.mark.anyio
async def test_cancellation_reaches_weather_chunk_reads(
    prepared_source: tuple[str, RecordingStore, list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip('fastapi')
    from anyio import CancelScope, CapacityLimiter, Event, create_task_group, fail_after

    source, store, _ = prepared_source
    variable = 'tas' if source == 'cmip6' else 'TMP'
    original_get = store.get
    started, finished = Event(), Event()
    active = set()
    scope = CancelScope()
    limiter = CapacityLimiter(1)

    async def blocking_get(key: str, *args, **kwargs):
        if f'/{variable}/c/' not in key:
            return await original_get(key, *args, **kwargs)
        active.add(key)
        started.set()
        try:
            await Event().wait()
        finally:
            active.remove(key)

    monkeypatch.setattr(store, 'get', blocking_get)
    repository = make_repository(source, store, limiter)

    async def read():
        with scope:
            await repository.read(make_query([variable]))
            pytest.fail('Cancelled chunk reads must not return data')
        finished.set()

    with fail_after(1):
        async with create_task_group() as tasks:
            tasks.start_soon(read)
            await started.wait()
            assert active
            scope.cancel()
            await finished.wait()
    assert active == set()
    assert limiter.borrowed_tokens == 0


@pytest.mark.anyio
async def test_async_sources_preserve_invalid_date_errors(
    prepared_source: tuple[str, RecordingStore, list[str]],
):
    pytest.importorskip('fastapi')
    from weatheasy.error import CMIP6DateRangeError, DateRangeError, ERA5DateRangeError

    source, store, variables = prepared_source
    repository = make_repository(source, store)
    query = make_query(variables)
    query['begin'], query['end'] = query['end'], query['begin']
    store.read_keys.clear()
    with pytest.raises(DateRangeError):
        await repository.read(query)
    assert store.read_keys == [], 'Invalid date order must be rejected before storage access'
    if source != 'cfs2':
        query['begin'] = query['end'] = date(2101, 1, 1)
        error = ERA5DateRangeError if source == 'era5' else CMIP6DateRangeError
        with pytest.raises(error):
            await repository.read(query)


@pytest.mark.anyio
@pytest.mark.parametrize('issue', ['current', 'stale', 'missing'])
async def test_cfs2_forecast_correction_is_served_only_for_current_issue(issue: str):
    pytest.importorskip('fastapi')
    from anyio import to_thread

    source, store, _ = prepared_store('cfs2')
    if issue != 'missing':
        with synchronous.open_dataset(store, 'forecast/flx') as forecast:
            corrected = forecast[['PRATE']].rename({'PRATE': 'PRATE_corr'}).load()
        corrected['PRATE_corr'] = corrected['PRATE_corr'] * 0 + 900
        corrected.PRATE_corr.attrs['units'] = 'mm/day'
        write_batch(store, 'forecast/correction', prepare_dataset(corrected, crs='EPSG:4326'))
        issued = '2020-03-04' if issue == 'current' else '2020-03-03'
        set_group_attrs(store, 'forecast/correction', forecast_updated=issued)
    query = make_query(['PRATE_corr'])
    actual = await make_repository(source, store).read(query)
    forecast = [900, 900] if issue == 'current' else [np.nan, np.nan]
    np.testing.assert_allclose(actual, [[513, np.nan, *forecast]], rtol=1e-6)
    reference = await to_thread.run_sync(lambda: synchronous.get_cfs2_data(store=store, **query))
    np.testing.assert_allclose(actual, reference, rtol=1e-6)
