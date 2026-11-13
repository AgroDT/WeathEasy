"""Open ERA5-Land GRIB fields lazily with the standard xarray/cfgrib backend."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import rioxarray  # noqa: F401 - register the xarray rio accessor
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from weatheasy import const


if TYPE_CHECKING:
    from pathlib import Path


def read_grib(path: Path, required: set[str]) -> xr.Dataset:
    from cfgrib.xarray_plugin import CfGribBackend  # noqa: PLC0415

    fields = []
    # Separate variables have different heights/soil levels. Share one header
    # index across their backends, without retaining .idx files in the raw cache.
    with TemporaryDirectory(prefix='weatheasy-grib-') as index_dir:
        indexpath = f'{index_dir}/{{short_hash}}.idx'
        for name in sorted(required):
            with xr.open_dataset(
                path,
                engine=CfGribBackend,
                cache=False,
                backend_kwargs={
                    'indexpath': indexpath,
                    'filter_by_keys': {'cfVarName': name},
                    'time_dims': ('valid_time',),
                    'squeeze': False,
                    'read_keys': ['edition', 'radius'],
                    'cache_geo_coords': False,
                    'errors': 'raise',
                    'values_dtype': np.dtype('float64'),
                },
            ) as ds:
                field = ds[name].squeeze(
                    [
                        dim
                        for dim in ds[name].dims
                        if dim not in ('valid_time', 'latitude', 'longitude')
                    ],
                    drop=True,
                )
                fields.append(field.reset_coords(drop=True).rename(valid_time='time'))
    return merge_grib_fields(*fields).rio.write_crs(const.ERA5_CRS)


def merge_grib_fields(*fields: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """Keep spatial axes exact while representing absent timestamps as NaN."""
    times = np.unique(np.concatenate([field.time.to_numpy() for field in fields]))
    aligned = [
        _align_time(field, times)
        if isinstance(field, xr.DataArray)
        else xr.Dataset(
            {
                name: _align_time(variable, times) if 'time' in variable.dims else variable
                for name, variable in field.data_vars.items()
            },
            attrs=field.attrs,
        )
        for field in fields
    ]
    return xr.merge(aligned, join='exact', compat='equals')


def _align_time(field: xr.DataArray, times: np.ndarray) -> xr.DataArray:
    if np.array_equal(field.time, times):
        return field
    # xarray.reindex eagerly masks non-Dask backend arrays. Delay filling gaps
    # until selection so a missing hour never materializes a whole monthly field.
    data = indexing.LazilyIndexedArray(_TimeAlignedArray(field, times))
    coords = {name: coord for name, coord in field.coords.items() if 'time' not in coord.dims}
    coords['time'] = times
    return xr.DataArray(
        xr.Variable(field.dims, data, attrs=field.attrs), coords=coords, name=field.name
    )


class _TimeAlignedArray(BackendArray):
    def __init__(self, field: xr.DataArray, times: np.ndarray) -> None:
        self.field = field
        self.shape = (len(times), *field.shape[1:])
        self.dtype = field.dtype
        self.positions = np.full(len(times), -1, dtype=np.intp)
        self.positions[np.searchsorted(times, field.time.to_numpy())] = np.arange(
            field.sizes['time']
        )

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key: tuple) -> np.ndarray:
        positions = np.atleast_1d(self.positions[key[0]])
        present = positions >= 0
        selection = self.field.isel(time=positions[present], latitude=key[1], longitude=key[2])
        if present.all():
            values = selection.to_numpy()
        else:
            values = np.full((len(positions), *selection.shape[1:]), np.nan, dtype=self.dtype)
            if present.any():
                values[present] = selection.to_numpy()
        return values[0] if np.isscalar(key[0]) else values
