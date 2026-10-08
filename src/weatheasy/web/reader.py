"""Asynchronous point reads from the prepared WeathEasy Zarr schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from anyio import fail_after
from rasterio.coords import BoundingBox
from xarray.coding.times import decode_cf_datetime
from zarr.api.asynchronous import open_group

from weatheasy.error import CoordsError
from weatheasy.storage import DIMS, SCHEMA, StorageError, point_indices


if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from datetime import date

    from numpy.typing import NDArray
    from zarr.abc.store import Store
    from zarr.core.group import AsyncGroup

    from weatheasy.query import Coords


class PointReader:
    """Bound each Zarr metadata or array operation within the query's cancel scope."""

    def __init__(self, store: Store, operation_timeout: float) -> None:
        self.store = store
        self.operation_timeout = operation_timeout

    async def wait[T](self, operation: Awaitable[T]) -> T:
        with fail_after(self.operation_timeout):
            return await operation

    async def group(self, path: str = '') -> AsyncGroup:
        return await self.wait(
            open_group(self.store, path=path, mode='r', zarr_format=3, use_consolidated=False)
        )

    async def times(self, group: AsyncGroup) -> NDArray[np.datetime64]:
        if (
            group.attrs.get('storage_schema') != SCHEMA
            or group.attrs.get('write_state') != 'complete'
        ):
            msg = 'Group has no completed coordinate-aware dataset'
            raise StorageError(msg)
        array = await self.wait(group.get_array('time'))
        if array.shape != (group.attrs.get('committed_length'),):
            msg = 'Stored time length differs from the commit marker'
            raise StorageError(msg)
        values = np.asarray(await self.wait(array.getitem(slice(None))))
        return decode_cf_datetime(
            values, cast('str', array.attrs['units']), cast('str', array.attrs['calendar'])
        )

    async def point(
        self,
        group: AsyncGroup,
        first: date,
        last: date,
        coords: Coords,
        variables: Sequence[str],
        *,
        times: NDArray[np.datetime64] | None = None,
    ) -> NDArray[np.float32]:
        times = await self.times(group) if times is None else times
        axes = []
        for name in ('latitude', 'longitude'):
            array = await self.wait(group.get_array(name))
            axes.append(np.asarray(await self.wait(array.getitem(slice(None)))))
        latitudes, longitudes = axes
        try:
            latitude, longitude = point_indices(
                latitudes, longitudes, latitude=coords.latitude, longitude=coords.longitude
            )
        except StorageError as exc:
            bbox = BoundingBox(
                float(longitudes.min()),
                float(latitudes.min()),
                float(longitudes.max()),
                float(latitudes.max()),
            )
            raise CoordsError(coords, bbox) from exc
        dates = np.arange(np.datetime64(first), np.datetime64(last) + np.timedelta64(1, 'D'))
        indices = np.searchsorted(times, dates)
        matched = indices < len(times)
        matched[matched] &= times[indices[matched]] == dates[matched]
        result = np.full((len(variables), len(dates)), np.nan, np.float32)
        for row, name in enumerate(variables):
            array = await self.wait(group.get_array(name))
            if array.metadata.dimension_names != DIMS or array.shape != (
                len(times),
                len(latitudes),
                len(longitudes),
            ):
                msg = f'{name} must match coordinate dimensions {DIMS}'
                raise StorageError(msg)
            if matched.any():
                values = await self.wait(
                    array.get_orthogonal_selection((indices[matched], latitude, longitude))
                )
                result[row, matched] = np.asarray(values, dtype=np.float32)
        return result
