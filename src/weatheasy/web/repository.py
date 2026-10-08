from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import aclosing
from dataclasses import dataclass
from datetime import date
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from anyio import CapacityLimiter, fail_after

from .reader import PointReader, PointSelection
from weatheasy import const
from weatheasy.error import CFS2Error, CMIP6DateRangeError, ERA5DateRangeError
from weatheasy.query import check_date_range, daily_precipitation


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .models import DataQuery


class StorageReadTimeoutError(TimeoutError):
    """The storage operation or query exceeded its budget."""


@dataclass
class ReadRegion:
    selection: PointSelection
    rows: list[int]
    first: date
    last: date


@dataclass
class ReadPlan:
    """Prepared regions assembled into chronological batches for one query."""

    variables: list[str]
    regions: list[ReadRegion]
    precipitation: str = 'PRATE'
    precipitation_factor: int = 86400

    async def read(self, first: date, last: date, /) -> NDArray[np.float32]:
        result = np.full((len(self.variables), (last - first).days + 1), np.nan, np.float32)
        for region in self.regions:
            begin, end = max(first, region.first), min(last, region.last)
            if begin > end:
                continue
            values = await region.selection.read(begin, end)
            offset = (begin - first).days
            result[region.rows, offset : offset + values.shape[1]] = values
        return daily_precipitation(
            result, self.variables, self.precipitation, self.precipitation_factor
        )


class WeatherRepository(ABC):
    """Read a ready store with the application's shared storage read budget."""

    __slots__ = ('_reader', 'read_limiter', 'read_timeout')

    def __init__(
        self,
        store: Store,
        read_limiter: CapacityLimiter,
        *,
        read_timeout: float = 60,
        operation_timeout: float = 10,
    ) -> None:
        self._reader = PointReader(store, operation_timeout)
        self.read_limiter = read_limiter
        self.read_timeout = read_timeout

    async def read(self, query: DataQuery) -> NDArray[np.float32]:
        """Collect batches for callers that need an in-memory result."""
        return np.concatenate([values async for values in self.stream(query)], axis=1)

    async def stream(
        self, query: DataQuery, *, batch_days: int = 64
    ) -> AsyncGenerator[NDArray[np.float32]]:
        """Read bounded batches, releasing the slot before yielding to the sender."""
        check_date_range(query['begin'], query['end'])
        if batch_days < 1:
            msg = 'batch_days must be positive'
            raise ValueError(msg)
        remaining = self.read_timeout
        plan = None
        days = (query['end'] - query['begin']).days + 1
        for offset in range(0, days, batch_days):
            first = query['begin'] + const.ONE_DAY * offset
            last = query['begin'] + const.ONE_DAY * (min(offset + batch_days, days) - 1)
            started = monotonic()
            try:
                if remaining <= 0:
                    raise StorageReadTimeoutError
                with fail_after(remaining):
                    async with self.read_limiter:
                        if plan is None:
                            plan = await self._prepare(query)
                        values = await plan.read(first, last)
            except TimeoutError as exc:
                raise StorageReadTimeoutError from exc
            remaining -= monotonic() - started
            if remaining <= 0:
                raise StorageReadTimeoutError
            yield values

    @abstractmethod
    async def _prepare(self, query: DataQuery, /) -> ReadPlan:
        """Validate the query and load metadata under the shared read budget."""


class Era5Repository(WeatherRepository):
    """Read prepared ERA5 weather data."""

    __slots__ = ()

    async def _prepare(self, query: DataQuery, /) -> ReadPlan:
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        reader = self._reader
        daily = await reader.group('daily')
        times = await reader.times(daily)
        first, last = (
            date.fromisoformat(str(value.astype('datetime64[D]'))) for value in times[[0, -1]]
        )
        for value in (begin, end):
            if not first <= value <= last:
                raise ERA5DateRangeError(value, first, last)
        regions = []
        for path, selected in (
            ('daily', [name for name in variables if name not in ('TMP_corr', 'PRATE_corr')]),
            ('correction', [name for name in variables if name in ('TMP_corr', 'PRATE_corr')]),
        ):
            if selected:
                group = daily if path == 'daily' else await reader.group(path)
                selection = await reader.prepare(
                    group, coords, selected, times=times if path == 'daily' else None
                )
                regions.append(
                    ReadRegion(selection, [variables.index(name) for name in selected], begin, end)
                )
        return ReadPlan(variables, regions)


class Cmip6Repository(WeatherRepository):
    """Read prepared CMIP6 weather data."""

    __slots__ = ()

    async def _prepare(self, query: DataQuery, /) -> ReadPlan:
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        first, last = date(const.CMIP6_FIRST_YEAR, 1, 1), date(const.CMIP6_LAST_YEAR, 12, 31)
        for value in (begin, end):
            if not first <= value <= last:
                raise CMIP6DateRangeError(value, first, last)
        reader = self._reader
        regions = []
        for row, name in enumerate(variables):
            group = await reader.group(name)
            selection = await reader.prepare(group, coords, [name])
            regions.append(ReadRegion(selection, [row], begin, end))
        return ReadPlan(variables, regions, precipitation='pr')


class Cfs2Repository(WeatherRepository):
    """Read prepared CFS2 weather data."""

    __slots__ = ()

    async def _prepare(self, query: DataQuery, /) -> ReadPlan:
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        reader = self._reader
        root = await reader.group()
        try:
            updated = root.attrs[const.CFS2_KEY_UPDATED]
        except KeyError as exc:
            raise CFS2Error from exc
        boundary = date.fromisoformat(str(updated)) - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
        regions = []
        for kind, first, last in (
            ('reanalysis', begin, min(end, boundary - const.ONE_DAY)),
            ('forecast', max(begin, boundary), end),
        ):
            if first > last:
                continue
            for grid, bands in (('flx', const.CFS2_FLX_BANDS), ('pgb', const.CFS2_PGB_BANDS)):
                selected = [name for name in variables if name in bands]
                if not selected:
                    continue
                group = await reader.group(f'{kind}/{grid}')
                if kind == 'reanalysis' and not group.attrs:
                    # Only empty loader-created groups mean absent reanalysis.
                    with fail_after(reader.operation_timeout):
                        async with aclosing(group.members()) as members:
                            if await anext(members, None) is None:
                                continue
                selection = await reader.prepare(group, coords, selected)
                regions.append(
                    ReadRegion(selection, [variables.index(name) for name in selected], first, last)
                )
            if kind == 'reanalysis':
                selected = [name for name in variables if name in ('TMP_corr', 'PRATE_corr')]
                if selected:
                    group = await reader.group('reanalysis/correction')
                    selection = await reader.prepare(group, coords, selected)
                    regions.append(
                        ReadRegion(
                            selection, [variables.index(name) for name in selected], first, last
                        )
                    )
        return ReadPlan(variables, regions, precipitation_factor=21600)
