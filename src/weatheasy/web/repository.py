from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import aclosing
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from anyio import CapacityLimiter, fail_after

from .reader import PointReader
from weatheasy import const
from weatheasy.error import CFS2Error, CMIP6DateRangeError, ERA5DateRangeError
from weatheasy.query import check_date_range, daily_precipitation


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .models import DataQuery


class StorageReadTimeoutError(TimeoutError):
    """The storage operation or query exceeded its budget."""


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
        try:
            with fail_after(self.read_timeout):
                async with self.read_limiter:
                    return await self._read(query)
        except TimeoutError as exc:
            raise StorageReadTimeoutError from exc

    @abstractmethod
    async def _read(self, query: DataQuery, /) -> NDArray[np.float32]:
        """Read source data inside the shared limiter and query timeout."""


class Era5Repository(WeatherRepository):
    """Read prepared ERA5 weather data."""

    __slots__ = ()

    async def _read(self, query: DataQuery, /) -> NDArray[np.float32]:
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        check_date_range(begin, end)
        reader = self._reader
        daily = await reader.group('daily')
        times = await reader.times(daily)
        first, last = (
            date.fromisoformat(str(value.astype('datetime64[D]'))) for value in times[[0, -1]]
        )
        for value in (begin, end):
            if not first <= value <= last:
                raise ERA5DateRangeError(value, first, last)
        result = np.full((len(variables), (end - begin).days + 1), np.nan, np.float32)
        for path, selected in (
            ('daily', [name for name in variables if name not in ('TMP_corr', 'PRATE_corr')]),
            ('correction', [name for name in variables if name in ('TMP_corr', 'PRATE_corr')]),
        ):
            if selected:
                group = daily if path == 'daily' else await reader.group(path)
                values = await reader.point(
                    group, begin, end, coords, selected, times=times if path == 'daily' else None
                )
                for row, name in enumerate(selected):
                    result[variables.index(name)] = values[row]
        return daily_precipitation(result, variables, 'PRATE', 86400)


class Cmip6Repository(WeatherRepository):
    """Read prepared CMIP6 weather data."""

    __slots__ = ()

    async def _read(self, query: DataQuery, /) -> NDArray[np.float32]:
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        check_date_range(begin, end)
        first, last = date(const.CMIP6_FIRST_YEAR, 1, 1), date(const.CMIP6_LAST_YEAR, 12, 31)
        for value in (begin, end):
            if not first <= value <= last:
                raise CMIP6DateRangeError(value, first, last)
        reader = self._reader
        rows = []
        for name in variables:
            group = await reader.group(name)
            rows.append((await reader.point(group, begin, end, coords, [name]))[0])
        return daily_precipitation(np.stack(rows), variables, 'pr', 86400)


class Cfs2Repository(WeatherRepository):
    """Read prepared CFS2 weather data."""

    __slots__ = ()

    async def _read(self, query: DataQuery, /) -> NDArray[np.float32]:  # noqa: C901
        begin, end = query['begin'], query['end']
        coords, variables = query['coords'], query['variables']
        check_date_range(begin, end)
        reader = self._reader
        root = await reader.group()
        try:
            updated = root.attrs[const.CFS2_KEY_UPDATED]
        except KeyError as exc:
            raise CFS2Error from exc
        boundary = date.fromisoformat(str(updated)) - const.CFS2_REANALYSIS_LAST_DATE_OFFSET
        result = np.full((len(variables), (end - begin).days + 1), np.nan, np.float32)
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
                values = await reader.point(group, first, last, coords, selected)
                offset = (first - begin).days
                for row, name in enumerate(selected):
                    result[variables.index(name), offset : offset + values.shape[1]] = values[row]
            if kind == 'reanalysis':
                selected = [name for name in variables if name in ('TMP_corr', 'PRATE_corr')]
                if selected:
                    group = await reader.group('reanalysis/correction')
                    values = await reader.point(group, first, last, coords, selected)
                    offset = (first - begin).days
                    for row, name in enumerate(selected):
                        result[variables.index(name), offset : offset + values.shape[1]] = values[
                            row
                        ]
        return daily_precipitation(result, variables, 'PRATE', 21600)
