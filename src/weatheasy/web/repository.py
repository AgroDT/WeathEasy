from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from anyio import CapacityLimiter, to_thread


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .models import DataQuery

    type Getter = Callable[..., NDArray[np.float32]]


class WeatherRepository:
    """Read a ready store with the application's shared storage read budget."""

    __slots__ = ('_getter', '_store', 'read_limiter')

    def __init__(self, store: Store, getter: Getter, read_limiter: CapacityLimiter) -> None:
        self._store = store
        self._getter = getter
        self.read_limiter = read_limiter

    async def read(self, query: DataQuery) -> NDArray[np.float32]:
        getter = partial(self._getter, **query, store=self._store)
        return await to_thread.run_sync(getter, limiter=self.read_limiter)
