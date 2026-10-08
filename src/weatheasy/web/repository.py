from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from anyio import CapacityLimiter, to_thread


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from .config import SourceSettings
    from .models import DataQuery

    type Getter = Callable[..., NDArray[np.float32]]


class WeatherRepository:
    """Read one source lazily with the application's shared storage read budget."""

    __slots__ = ('_getter', '_source', 'read_limiter')

    def __init__(
        self, source: SourceSettings, getter: Getter, read_limiter: CapacityLimiter
    ) -> None:
        self._source = source
        self._getter = getter
        self.read_limiter = read_limiter

    async def read(self, query: DataQuery) -> NDArray[np.float32]:
        getter = partial(self._getter, **query, store=self._source.zarr_store)
        return await to_thread.run_sync(getter, limiter=self.read_limiter)
