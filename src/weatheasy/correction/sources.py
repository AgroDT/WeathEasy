"""Coordinate-aware inputs for offline reanalysis bias correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from weatheasy.storage import open_dataset


if TYPE_CHECKING:
    from datetime import date

    import xarray as xr
    import zarr
    from numpy.typing import NDArray


RUSSIA_BBOX = (82, 19, 41, 180)  # north, west, south, east


@dataclass(frozen=True)
class Source:
    """Raw grid paths, correction region, and precipitation rate conversion."""

    name: str
    group_path: str
    correction_path: str
    temp_var: str = 'TMP'
    precip_var: str = 'PRATE'
    precip_to_mm: float = 86400.0
    crop_bbox: tuple[int, int, int, int] | None = None

    def path(self, root: zarr.Group) -> str:
        return '/'.join(filter(None, (root.path, self.group_path)))

    def corrected_path(self, root: zarr.Group) -> str:
        return '/'.join(filter(None, (root.path, self.correction_path)))

    def grid(self, root: zarr.Group) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        with open_dataset(root.store, self.path(root)) as ds:
            view = self._crop(ds)
            return (
                view.latitude.to_numpy().astype(np.float32),
                view.longitude.to_numpy().astype(np.float32),
            )

    def read_day(
        self,
        root: zarr.Group,
        var: str,
        day: date,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Read exactly one committed day and return its actual coordinate axes."""
        with open_dataset(root.store, self.path(root)) as ds:
            view = self._crop(ds)
            stamp = np.datetime64(day)
            if stamp not in view.time.to_numpy().astype('datetime64[D]'):
                msg = f'{self.name}: {day} is not in the prepared reanalysis'
                raise IndexError(msg)
            grid = view[var].sel(time=stamp).to_numpy().astype(np.float32)
            return (
                grid,
                view.latitude.to_numpy().astype(np.float32),
                view.longitude.to_numpy().astype(np.float32),
            )

    def _crop(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.crop_bbox is None:
            return dataset
        north, west, south, east = self.crop_bbox
        return dataset.sel(latitude=slice(south, north), longitude=slice(west, east))


ERA5 = Source('era5', 'daily', 'correction')
CFS2 = Source(
    'cfs2',
    'reanalysis/flx',
    'reanalysis/correction',
    precip_to_mm=21600.0,
    crop_bbox=RUSSIA_BBOX,
)
SOURCES: dict[str, Source] = {ERA5.name: ERA5, CFS2.name: CFS2}
