"""Coordinate-aware, single-writer Zarr v3 storage.

Each source owns a separate store. CFS2 grids use reanalysis/{flx,pgb} and
forecast/{flx,pgb}; ERA5 uses daily; CMIP6 uses <variable>.
A group contains (time, latitude, longitude) fields, true cell centres and a
scalar spatial_ref with the source CRS. Latitude increases; longitude increases
in a continuous, unwrapped interval (e.g. 170..190 across the dateline). No
resampling, unit conversion or affine approximation of Gaussian axes occurs.
Time is the UTC valid timestamp, without timezone, encoded as Gregorian seconds.
Units and temporal cell_methods belong to each field, supplied by its producer.

Only write_state='complete' is publishable, including for direct xarray users.
One external writer owns a group. Before appending into a partially filled time
chunk, its committed objects are saved durably. Retry restores these objects
before truncating the uncommitted tail and writing again. Concurrent
writers, overwriting committed dates and old Zarr layouts are not supported.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import rioxarray
import xarray as xr
import zarr
from xarray.coding.times import decode_cf_datetime
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore

from weatheasy.error import S3ImportError


if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from mmap import mmap
    from os import PathLike

    from fsspec.asyn import AsyncFileSystem
    from rasterio import CRS
    from zarr.abc.store import Store
    from zarr.core.metadata import ArrayV3Metadata

SCHEMA = 'weatheasy-coordinate-v1'
DIMS = ('time', 'latitude', 'longitude')


class StorageError(ValueError):
    """Invalid schema, conflicting grid or incomplete write."""


def get_store(uri: str | PathLike[str], *, filesystem: AsyncFileSystem | None = None) -> Store:
    """Create a Zarr store from a local path or S3 URI.

    Pass one asynchronous S3 filesystem to reuse its connection pool across
    stores within a job. Local paths need no S3 dependencies or filesystem.
    """
    location = str(uri)
    if location.startswith('s3://'):
        if not location[5:].split('/', maxsplit=1)[0]:
            msg = 'S3 URI requires a bucket'
            raise StorageError(msg)
        if filesystem is None:
            try:
                import s3fs  # noqa: PLC0415
            except ImportError as exc:
                raise S3ImportError from exc
            filesystem = s3fs.S3FileSystem(asynchronous=True)
        return FsspecStore(filesystem, path=location[5:])
    if filesystem is not None:
        msg = 'An S3 filesystem can only be used with an s3:// URI'
        raise StorageError(msg)
    return LocalStore(location)


def group_attrs(store: Store, group: str) -> dict[str, Any]:
    """Read metadata from a group without exposing Zarr objects to readers."""
    root = zarr.open_group(store, mode='r', zarr_format=3, use_consolidated=False)
    target = root.get_group(group) if group else root
    return dict(target.attrs)


def group_exists(store: Store, group: str) -> bool:
    """Check whether a group exists without creating it."""
    try:
        root = zarr.open_group(store, mode='r', zarr_format=3, use_consolidated=False)
    except FileNotFoundError:
        return False
    if not group:
        return True
    if group not in root:
        return False
    if not isinstance(root[group], zarr.Group):
        msg = f'{group} must be a Zarr group'
        raise TypeError(msg)
    return True


def pristine_group(store: Store, group: str) -> bool:
    """Recognize an empty loader-created group; incomplete writes are not empty."""
    root = zarr.open_group(store, mode='r', zarr_format=3, use_consolidated=False)
    target = root.get_group(group)
    return not target.attrs and not any(target.keys())


def set_group_attrs(store: Store, group: str, **attrs: Any) -> None:
    """Create a metadata group if needed and update its attributes."""
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
    root.update_attributes(dict(root.attrs))
    target = root.require_group(group) if group else root
    target.attrs.update(attrs)
    zarr.consolidate_metadata(store)


def remove_group_attr(store: Store, group: str, key: str) -> None:
    """Remove a readiness marker before mutating its published data."""
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
    root.update_attributes(dict(root.attrs))
    target = root.get_group(group) if group else root
    target.attrs.pop(key, None)
    zarr.consolidate_metadata(store)


def delete_group(store: Store, group: str) -> None:
    """Remove a group in place, retaining the current published path."""
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
    if group in root:
        root.update_attributes(dict(root.attrs))
        del root[group]
        zarr.consolidate_metadata(store)


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
        ds = ds.assign_coords({axis: values})
        if np.any(np.diff(values) < 0):
            ds = ds.sortby(axis)
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
    lat_index, lon_index = point_indices(
        dataset.latitude.to_numpy(),
        dataset.longitude.to_numpy(),
        latitude=latitude,
        longitude=longitude,
    )
    return dataset.isel(latitude=lat_index, longitude=lon_index)


def point_indices(
    latitudes: np.ndarray, longitudes: np.ndarray, *, latitude: float, longitude: float
) -> tuple[int, int]:
    """Validate a point and find its nearest centres on normalized increasing axes."""
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        msg = 'Coordinates must be finite'
        raise StorageError(msg)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 360:
        msg = 'Invalid geographic coordinates'
        raise StorageError(msg)
    south, north = _edges(latitudes)
    west, east = _edges(longitudes)
    wrapped = (longitude - west) % 360 + west
    if not max(-90, south) <= latitude <= min(90, north) or wrapped > east + 1e-10:
        msg = 'Point is outside this grid'
        raise StorageError(msg)

    def nearest(values: np.ndarray, value: float) -> int:
        distances = abs(values - value)
        # Increasing axes: the last nearest centre wins north/east ties.
        return int(np.flatnonzero(distances == distances.min())[-1])

    return nearest(latitudes, latitude), nearest(longitudes, wrapped)


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


async def _keys(store: Store, prefix: str) -> list[str]:
    return [key async for key in store.list_prefix(prefix)]


def _restore_append_backup(store: Store, target: zarr.Group) -> None:
    # The marker is retained until commit, so restoration itself is retryable.
    if 'append_backup' not in target.attrs:
        return
    count = cast('int', target.attrs['committed_length'])
    for _, array in target.arrays():
        dimensions = cast('ArrayV3Metadata', array.metadata).dimension_names
        if dimensions and 'time' in dimensions:
            chunk = array.chunks[dimensions.index('time')]
            if count % chunk:
                sync(store.delete_dir(f'{array.path}/c/{count // chunk}/'))
    for key in cast('list[str]', target.attrs['append_backup']):
        saved = sync(store.get(f'{target.path}/.append-backup/{key}', default_buffer_prototype()))
        if saved is None:
            msg = f'Missing committed chunk backup: {key}'
            raise StorageError(msg)
        sync(store.set(key, saved))


def _save_append_backup(store: Store, target: zarr.Group, count: int) -> None:
    keys = []
    for _, array in target.arrays():
        dimensions = cast('ArrayV3Metadata', array.metadata).dimension_names
        if not dimensions or 'time' not in dimensions:
            continue
        axis = dimensions.index('time')
        chunk = array.chunks[axis]
        if count % chunk == 0:
            continue
        # Weather time is the leading dimension. Coordinates use time chunks
        # of one, so their committed objects are never touched by append.
        prefix = f'{array.path}/c/{count // chunk}/'
        for key in sync(_keys(store, prefix)):
            value = sync(store.get(key, default_buffer_prototype()))
            if value is not None:
                sync(store.set(f'{target.path}/.append-backup/{key}', value))
                keys.append(key)
    # Publish the manifest only after every backup is durable, before mutation.
    target.attrs['append_backup'] = keys


def write_batch(
    store: Store,
    group: str,
    dataset: xr.Dataset,
    *,
    tile_size: int = 32,
    time_chunk: int | None = None,
) -> None:
    """Create or append prepared data, publishing only after all arrays are written.

    The caller serializes writers. A failed write is retried with the same grid
    and field metadata; previously committed dates must not be supplied again.
    The caller supplies a ready local or S3 store. Local memory-mapped staging
    feeds chunk-aligned Zarr writes; recovery restores the committed boundary.
    """
    write_batches(store, group, [dataset], tile_size=tile_size, time_chunk=time_chunk)


def write_batches(  # noqa: C901, PLR0912, PLR0915
    store: Store,
    group: str,
    datasets: Iterable[xr.Dataset],
    *,
    tile_size: int = 32,
    time_chunk: int | None = None,
) -> None:
    """Stage bounded reductions on local disk, then write aligned weather chunks.

    Staging avoids keeping an entire weather month in RAM and prevents repeated
    remote read/modify/write for each day within a long time chunk.
    """
    time_chunk = (64 if 'forecast/' in group else 128) if time_chunk is None else time_chunk
    batches = iter(datasets)
    dataset = next(batches, None)
    if dataset is None:
        return
    if tile_size < 1 or time_chunk < 1 or dataset.attrs.get('storage_schema') != SCHEMA:
        msg = 'A prepared dataset and positive chunk sizes are required'
        raise StorageError(msg)
    signature = _signature(dataset)
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
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
            time_array = target.get_array('time')
            last = decode_cf_datetime(
                time_array[count - 1 : count],
                cast('str', time_array.attrs['units']),
                cast('str', time_array.attrs['calendar']),
            )[0]
            if dataset.time.to_numpy()[0] <= last:
                msg = 'New timestamps must follow all committed timestamps'
                raise StorageError(msg)
    elif list(target.keys()) or target.attrs:
        msg = 'Existing group is not a coordinate-aware weatheasy store'
        raise StorageError(msg)
    root.update_attributes(dict(root.attrs))
    attrs = dict(
        dataset.attrs, grid_signature=signature, committed_length=count, write_state='writing'
    )
    # Preserve an interrupted append's manifest until restoration is complete.
    target.attrs.update(attrs)
    with TemporaryDirectory(prefix='weatheasy-weather-') as directory:
        first_dataset = dataset.isel(time=slice(0, 1)).copy(deep=True)
        field_names = ['time', *(str(name) for name in dataset.data_vars)]
        filenames = {
            name: Path(directory) / f'{index}.raw' for index, name in enumerate(field_names)
        }
        written = 0
        previous = None
        while dataset is not None:
            if dataset.attrs.get('storage_schema') != SCHEMA or _signature(dataset) != signature:
                msg = 'A batch has a conflicting grid, CRS, variables or units'
                raise StorageError(msg)
            if previous is not None and dataset.time.to_numpy()[0] <= previous:
                msg = 'New timestamps must follow all previous batches'
                raise StorageError(msg)
            for name, filename in filenames.items():
                values = (
                    dataset.time.to_numpy().astype('datetime64[s]').astype(np.int64)
                    if name == 'time'
                    else dataset[name].to_numpy()
                )
                with filename.open('ab') as stream:
                    values.tofile(stream)
            written += dataset.sizes['time']
            previous = dataset.time.to_numpy()[-1]
            dataset = next(batches, None)
        if count:
            _restore_append_backup(store, target)
            _truncate_uncommitted(target, count)
            _save_append_backup(store, target, count)
        else:
            first_dataset.attrs = attrs
            _write_time_batch(
                store, group, first_dataset, tile_size=tile_size, time_chunk=time_chunk
            )
            target = root.get_group(group)
        staged = {
            name: np.memmap(
                filename,
                mode='r',
                dtype=np.int64 if name == 'time' else first_dataset[name].dtype,
                shape=(written,)
                if name == 'time'
                else (written, first_dataset.sizes['latitude'], first_dataset.sizes['longitude']),
            )
            for name, filename in filenames.items()
        }
        _copy_staged(target, staged, count, written)
        del staged
    final_attrs = dict(target.attrs)
    final_attrs.pop('append_backup', None)
    final_attrs.update(committed_length=count + written, write_state='complete')
    target.attrs.put(final_attrs)
    zarr.consolidate_metadata(store)
    sync(store.delete_dir(f'{group}/.append-backup/'))


def write_blocks(
    store: Store,
    group: str,
    datasets: Iterable[xr.Dataset],
    *,
    work_dir: str | Path,
    time_chunk: int = 128,
) -> None:
    """Stage up to 128 daily fields in the working directory, then write full chunks."""
    log = logging.getLogger(__name__)
    directory = Path(work_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
    target = root.require_group(group)
    count = int(cast('int', target.attrs.get('committed_length', 0)))
    days = iter(datasets)
    while (first := next(days, None)) is not None:
        if count % time_chunk:
            msg = 'Cannot append to a partial time chunk'
            raise StorageError(msg)
        attrs = dict(
            first.attrs,
            grid_signature=_signature(first),
            committed_length=count,
            write_state='writing',
        )
        with TemporaryDirectory(prefix='weatheasy-weather-', dir=directory) as temporary:
            template = first.isel(time=slice(0, 0)).copy()
            template.attrs = attrs
            filenames = {
                name: Path(temporary) / f'{index}.raw'
                for index, name in enumerate(['time', *(str(name) for name in first.data_vars)])
            }
            log.info('Staging %s: up to 128 days in %s', group, temporary)
            written = 0
            for day in chain((first,), islice(days, 127)):
                for name, filename in filenames.items():
                    values = (
                        day.time.to_numpy().astype('datetime64[s]').astype(np.int64)
                        if name == 'time'
                        else day[name].to_numpy()
                    )
                    with filename.open('ab') as stream:
                        values.tofile(stream)
                written += day.sizes['time']
            del first, day, values
            log.info('Writing %s: %s days', group, written)
            root.update_attributes(dict(root.attrs))
            target.attrs.update(attrs)
            if not count:
                _write_time_batch(store, group, template, tile_size=32, time_chunk=time_chunk)
                target = root.get_group(group)
            for name, filename in filenames.items():
                staged = np.memmap(
                    filename,
                    mode='r',
                    dtype=np.int64 if name == 'time' else template[name].dtype,
                    shape=(written,)
                    if name == 'time'
                    else (written, template.sizes['latitude'], template.sizes['longitude']),
                )
                try:
                    _copy_staged(target, {name: staged}, count, written)
                finally:
                    cast('mmap', staged.base).close()
                    del staged
        count += written
        target.attrs.update(dict(attrs, committed_length=count, write_state='complete'))
        zarr.consolidate_metadata(store)
        log.info('Saved %s: %s committed days', group, count)


def _copy_staged(
    target: zarr.Group, staged: dict[str, np.memmap], count: int, written: int
) -> None:
    for name, source_array in staged.items():
        output = target.get_array(name)
        output.resize((count + written, *output.shape[1:]))
        step = output.chunks[0]
        for chunk_index in range(count // step, (count + written - 1) // step + 1):
            first = max(count, chunk_index * step)
            stop = min((chunk_index + 1) * step, count + written)
            # Align each destination chunk, including a partial committed chunk.
            if first < stop:
                source_time = slice(first - count, stop - count)
                if output.ndim == 1:
                    output[first:stop] = source_array[source_time]
                else:
                    for lat in range(0, output.shape[1], output.chunks[1]):
                        for lon in range(0, output.shape[2], output.chunks[2]):
                            spatial = (
                                slice(lat, lat + output.chunks[1]),
                                slice(lon, lon + output.chunks[2]),
                            )
                            output[(slice(first, stop), *spatial)] = source_array[
                                (source_time, *spatial)
                            ]


def _write_time_batch(
    store: Store, group: str, ds: xr.Dataset, *, tile_size: int, time_chunk: int
) -> None:
    encoding: dict[Hashable, dict[str, Any]] = {
        name: {
            'chunks': (
                time_chunk,
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


def open_dataset(store: Store, group: str) -> xr.Dataset:
    """Open a fully committed group without GRIB ingestion dependencies or Dask."""
    root = zarr.open_group(store, mode='r', zarr_format=3, use_consolidated=False)
    target = root.get_group(group)
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


def last_date(store: Store, group: str) -> date | None:
    """Last committed day, including after a failed append awaiting retry."""
    root = zarr.open_group(store, mode='a', zarr_format=3, use_consolidated=False)
    if group not in root:
        return None
    target = root.get_group(group)
    count = int(cast('int', target.attrs.get('committed_length', 0)))
    if not count:
        return None
    times = target.get_array('time')
    value = decode_cf_datetime(
        times[count - 1 : count],
        cast('str', times.attrs['units']),
        cast('str', times.attrs['calendar']),
    )[0]
    return date.fromisoformat(str(value.astype('datetime64[D]')))
