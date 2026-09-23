"""Coordinate-aware, single-writer Zarr v3 storage.

Each grid is a separate group: cfs2/reanalysis/{flx,pgb},
cfs2/forecast/{flx,pgb}, or cmip6/<variable>.
A group contains (time, latitude, longitude) fields, true cell centres and a
scalar spatial_ref with the source CRS. Latitude increases; longitude increases
in a continuous, unwrapped interval (e.g. 170..190 across the dateline). No
resampling, unit conversion or affine approximation of Gaussian axes occurs.
Time is the UTC valid timestamp, without timezone, encoded as Gregorian seconds.
Units and temporal cell_methods belong to each field, supplied by its producer.

Only write_state='complete' is publishable, including for direct xarray users.
One external writer owns a group. Appends use time chunks of one so a failed
append cannot damage a chunk of previously committed data. A retry truncates
only the uncommitted tail of this schema before writing again. Concurrent
writers, overwriting committed dates and old Zarr layouts are not supported.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import rioxarray
import xarray as xr
import zarr


if TYPE_CHECKING:
    from rasterio.crs import CRS
    from zarr.core.metadata import ArrayV3Metadata
    from zarr.storage import StoreLike

SCHEMA = 'weatheasy-coordinate-v1'
DIMS = ('time', 'latitude', 'longitude')


class StorageError(ValueError):
    """Invalid schema, conflicting grid or incomplete write."""


def _normalize_spatial_axes(ds: xr.Dataset) -> xr.Dataset:
    for axis in ('latitude', 'longitude'):
        values = np.asarray(ds[axis], dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            msg = f'{axis} requires at least two finite cell centres'
            raise StorageError(msg)
        if axis == 'longitude':
            values = np.unwrap(values, period=360)
            values -= 360 * np.floor((values.min() + 180) / 360)
        elif np.any(abs(values) > 90):
            msg = 'Latitude centres must be within [-90, 90]'
            raise StorageError(msg)
        ds = ds.assign_coords({axis: values}).sortby(axis)
        if np.any(np.diff(ds[axis].to_numpy()) <= 0):
            msg = f'Duplicate {axis} centres'
            raise StorageError(msg)
    return ds


def prepare_dataset(dataset: xr.Dataset, *, crs: str | CRS) -> xr.Dataset:
    """Normalize axes, retaining their actual centres and associated values."""
    ds = dataset.copy()
    if set(ds.dims) != set(DIMS):
        msg = f'Expected dimensions {DIMS}, got {tuple(ds.dims)}'
        raise StorageError(msg)
    ds = _normalize_spatial_axes(ds)
    times = np.asarray(ds.time)
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        msg = 'time must contain Gregorian UTC datetime64 timestamps'
        raise StorageError(msg)
    if not times.size or np.any(np.diff(times) <= np.timedelta64(0, 's')):
        msg = 'time must be nonempty, unique and increasing'
        raise StorageError(msg)
    ds = ds.assign_coords(time=times.astype('datetime64[ns]')).transpose(*DIMS)
    fields = list(ds.data_vars)
    if not fields:
        msg = 'At least one weather variable is required'
        raise StorageError(msg)
    for name in fields:
        if ds[name].dims != DIMS or not np.issubdtype(ds[name].dtype, np.floating):
            msg = f'{name} must be a floating-point field with dimensions {DIMS}'
            raise StorageError(msg)
        if not ds[name].attrs.get('units'):
            msg = f'{name} requires explicit units'
            raise StorageError(msg)
        ds[name].encoding = {}
    ds.time.attrs = {'standard_name': 'time', 'axis': 'T'}
    # Drop inherited transform metadata: Gaussian latitude has no affine
    # transform. rio writes CF CRS/axis metadata without deriving a transform.
    ds = ds.drop_vars('spatial_ref', errors='ignore')
    ds.attrs = {'storage_schema': SCHEMA, 'Conventions': 'CF-1.8'}
    with rioxarray.set_options(export_grid_mapping=True):
        ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        ds = ds.rio.write_crs(crs, grid_mapping_name='spatial_ref')
        ds = ds.rio.write_coordinate_system()
    west, east = _edges(ds.longitude.to_numpy())
    if east - west > 360 + 1e-6:
        msg = 'Longitude coverage exceeds one revolution'
        raise StorageError(msg)
    return ds


def merge_fields(*datasets: xr.Dataset) -> xr.Dataset:
    """Combine fields only when all coordinates match; never align different grids."""
    return xr.merge(datasets, join='exact', compat='equals', combine_attrs='identical')


def _edges(values: np.ndarray) -> tuple[float, float]:
    return (
        float(values[0] - (values[1] - values[0]) / 2),
        float(values[-1] + (values[-1] - values[-2]) / 2),
    )


def select_point(dataset: xr.Dataset, *, latitude: float, longitude: float) -> xr.Dataset:
    """Nearest centre; reject points outside outer half-cells; ties go north/east.

    Longitude queries use [-180, 360]; periodic equivalents, including +/-180,
    select the same cell. Bounds come from adjacent centres, clipped at poles.
    On global seam ties the equivalent centre in the stored interval wins.
    """
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        msg = 'Coordinates must be finite'
        raise StorageError(msg)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 360:
        msg = 'Invalid geographic coordinates'
        raise StorageError(msg)
    south, north = _edges(dataset.latitude.to_numpy())
    west, east = _edges(dataset.longitude.to_numpy())
    wrapped = (longitude - west) % 360 + west
    if not max(-90, south) <= latitude <= min(90, north) or wrapped > east + 1e-10:
        msg = 'Point is outside this grid'
        raise StorageError(msg)
    return dataset.sel(latitude=latitude, longitude=wrapped, method='nearest')


def _signature(ds: xr.Dataset) -> str:
    description = {
        'crs': ds.spatial_ref.attrs['crs_wkt'],
        'fields': {
            name: (str(field.dtype), dict(field.attrs))
            for name, field in sorted(ds.data_vars.items())
        },
    }
    digest = hashlib.sha256(json.dumps(description, sort_keys=True, default=str).encode())
    for name in ('latitude', 'longitude'):
        digest.update(ds[name].to_numpy().astype('<f8').tobytes())
    return digest.hexdigest()


def _truncate_uncommitted(target: zarr.Group, count: int) -> None:
    for _, array in target.arrays():
        dimensions = cast('ArrayV3Metadata', array.metadata).dimension_names
        if dimensions and 'time' in dimensions:
            axis = dimensions.index('time')
            if array.shape[axis] < count:
                msg = 'A committed array is truncated; cannot recover'
                raise StorageError(msg)
            shape = list(array.shape)
            shape[axis] = count
            if tuple(shape) != array.shape:
                array.resize(tuple(shape))


def write_batch(store: StoreLike, group: str, dataset: xr.Dataset, *, tile_size: int = 128) -> None:
    """Create or append prepared data, publishing only after all arrays are written.

    The caller serializes writers. A failed write is retried with the same grid
    and field metadata; previously committed dates must not be supplied again.
    S3 callers may pass zarr.storage.FsspecStore with an asynchronous S3 filesystem.
    """
    if tile_size < 1 or dataset.attrs.get('storage_schema') != SCHEMA:
        msg = 'A prepared dataset and positive tile_size are required'
        raise StorageError(msg)
    ds = dataset.copy()
    signature = _signature(ds)
    root = zarr.open_group(store, mode='a', zarr_format=3)
    target = root.require_group(group)
    count = int(cast('int', target.attrs.get('committed_length', 0)))
    if 'storage_schema' in target.attrs:
        if (
            target.attrs['storage_schema'] != SCHEMA
            or target.attrs.get('grid_signature') != signature
        ):
            msg = 'Existing group has a conflicting grid, CRS, variables or units'
            raise StorageError(msg)
        if count:
            # Validate incoming dates before any recovery/mutation.
            time_array = cast('zarr.Array', target['time'])
            last = xr.coding.times.decode_cf_datetime(
                time_array[count - 1 : count],
                cast('str', time_array.attrs['units']),
                cast('str', time_array.attrs['calendar']),
            )[0]
            if ds.time.to_numpy()[0] <= last:
                msg = 'New timestamps must follow all committed timestamps'
                raise StorageError(msg)
            _truncate_uncommitted(target, count)
    elif list(target.keys()) or target.attrs:
        msg = 'Existing group is not a coordinate-aware weatheasy store'
        raise StorageError(msg)
    attrs = dict(ds.attrs, grid_signature=signature, committed_length=count, write_state='writing')
    target.attrs.update(attrs)
    ds.attrs = attrs
    if count:
        # a- leaves fixed spatial coordinates untouched.
        ds.to_zarr(
            store,
            group=group,
            mode='a-',
            append_dim='time',
            zarr_format=3,
            consolidated=False,
            write_empty_chunks=False,
        )
    else:
        encoding = {
            name: {
                'chunks': (
                    1,
                    min(tile_size, ds.sizes['latitude']),
                    min(tile_size, ds.sizes['longitude']),
                ),
                '_FillValue': np.nan,
                'fill_value': np.nan,
                'grid_mapping': 'spatial_ref',
            }
            for name in ds.data_vars
        }
        encoding['time'] = {
            'chunks': (1,),
            'dtype': 'int64',
            'units': 'seconds since 1970-01-01',
            'calendar': 'proleptic_gregorian',
        }
        for axis in ('latitude', 'longitude'):
            encoding[axis] = {'chunks': (ds.sizes[axis],), '_FillValue': None}
        ds.to_zarr(
            store,
            group=group,
            mode='w',
            encoding=encoding,
            zarr_format=3,
            consolidated=False,
            write_empty_chunks=False,
        )
    target = cast('zarr.Group', root[group])
    target.attrs.update(committed_length=count + ds.sizes['time'], write_state='complete')


def open_dataset(store: StoreLike, group: str) -> xr.Dataset:
    """Open a fully committed group without GRIB ingestion dependencies or Dask."""
    root = zarr.open_group(store, mode='r', zarr_format=3)
    target = cast('zarr.Group', root[group])
    if (
        target.attrs.get('storage_schema') != SCHEMA
        or target.attrs.get('write_state') != 'complete'
    ):
        msg = 'Group has no completed coordinate-aware dataset'
        raise StorageError(msg)
    ds = xr.open_zarr(
        store, group=group, consolidated=False, chunks=None, decode_coords='all', zarr_format=3
    )
    if ds.sizes['time'] != target.attrs['committed_length']:
        ds.close()
        msg = 'Stored time length differs from the commit marker'
        raise StorageError(msg)
    return ds


def last_date(store: StoreLike, group: str) -> date | None:
    """Last committed day, including after a failed append awaiting retry."""
    root = zarr.open_group(store, mode='a', zarr_format=3)
    if group not in root:
        return None
    target = cast('zarr.Group', root[group])
    count = int(cast('int', target.attrs.get('committed_length', 0)))
    if not count:
        return None
    times = cast('zarr.Array', target['time'])
    value = xr.coding.times.decode_cf_datetime(
        times[count - 1 : count],
        cast('str', times.attrs['units']),
        cast('str', times.attrs['calendar']),
    )[0]
    return date.fromisoformat(str(value.astype('datetime64[D]')))
