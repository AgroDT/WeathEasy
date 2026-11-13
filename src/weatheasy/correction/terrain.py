"""Prepare reusable weather terrain and native-resolution station features."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import rioxarray
import xarray as xr
import zarr
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from weatheasy.storage import point_indices


if TYPE_CHECKING:
    from collections.abc import Generator

    import pandas as pd
    from zarr.abc.store import Store

    from .sources import Source


@contextmanager
def _native(store: Store, elevation: str, x: str, y: str) -> Generator[xr.DataArray]:
    with xr.open_zarr(store, chunks=None, consolidated=None, decode_coords='all') as ds:
        field = ds[elevation].rio.set_spatial_dims(x_dim=x, y_dim=y)
        if field.rio.crs is None:
            field = field.rio.write_crs('EPSG:4326')
        chunks = dict(zip(field.dims, field.encoding.get('chunks', (1200, 1200)), strict=True))
        reverse = {}
        if field.rio.crs != 'EPSG:4326':
            msg = 'DEM preparation requires EPSG:4326'
            raise ValueError(msg)
        field = field.rename({x: 'longitude', y: 'latitude'}).transpose('latitude', 'longitude')
        for axis in ('latitude', 'longitude'):
            coords = field[axis].to_numpy()
            if len(coords) < 2 or not np.isfinite(coords).all():
                msg = f'DEM {axis} requires at least two finite coordinates'
                raise ValueError(msg)
            reverse[axis] = bool(np.all(np.diff(coords) < 0))
            if reverse[axis]:
                field = field.isel({axis: slice(None, None, -1)})
            elif not np.all(np.diff(coords) > 0):
                msg = f'DEM {axis} coordinates must be monotonic'
                raise ValueError(msg)
        field.encoding['chunks'] = (chunks[y], chunks[x])
        field.attrs['_source_reversed'] = reverse
        yield field


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.r_[
        values[0] - (values[1] - values[0]) / 2,
        (values[:-1] + values[1:]) / 2,
        values[-1] + (values[-1] - values[-2]) / 2,
    ]


def _slopes(elevation: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Signed surface derivatives in metres per metre, including Gaussian latitude."""
    north = np.gradient(elevation.astype(np.float64), np.deg2rad(lats) * 6_371_000, axis=0)
    east = np.gradient(elevation.astype(np.float64), np.deg2rad(lons) * 6_371_000, axis=1)
    east /= np.cos(np.deg2rad(lats))[:, None]
    return east.astype(np.float32), north.astype(np.float32)


def _weights(target_edges: np.ndarray, source_edges: np.ndarray, *, latitude: bool) -> np.ndarray:
    low = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    high = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    if latitude:
        low = np.sin(np.deg2rad(np.clip(low, -90, 90)))
        high = np.sin(np.deg2rad(np.clip(high, -90, 90)))
    return np.maximum(high - low, 0)


def _chunk_slices(field: xr.DataArray, axis: str, step: int) -> list[slice]:
    size = field.sizes[axis]
    offset = size % step if field.attrs['_source_reversed'][axis] else 0
    boundaries = sorted({0, size, *range(offset, size, step)})
    return [slice(start, stop) for start, stop in pairwise(boundaries)]


def _axis_blocks(
    field: xr.DataArray,
    axis: str,
    step: int,
    target: np.ndarray,
    source: np.ndarray,
    *,
    periodic: bool = False,
) -> list[tuple]:
    blocks = []
    shifts = (-360, 0, 360) if periodic else (0,)
    for selected in _chunk_slices(field, axis, step):
        pieces = []
        for shift in shifts:
            edges = source[selected.start : selected.stop + 1] + shift
            if edges[-1] <= target[0] or edges[0] >= target[-1]:
                continue
            first = max(0, int(np.searchsorted(target, edges[0], side='right')) - 1)
            stop = min(len(target) - 1, int(np.searchsorted(target, edges[-1], side='left')))
            pieces.append((first, stop, edges))
        if not pieces:
            continue
        first, stop = min(p[0] for p in pieces), max(p[1] for p in pieces)
        weights = sum(
            (
                _weights(target[first : stop + 1], edges, latitude=axis == 'latitude')
                for _, _, edges in pieces
            )
        )
        blocks.append((selected, np.arange(first, stop), weights))
    return blocks


def _aggregate(field: xr.DataArray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Area-weighted cell averages; each intersecting source chunk is read once."""
    target_y, target_x = _edges(lats), _edges(lons)
    source_y, source_x = _edges(field.latitude.to_numpy()), _edges(field.longitude.to_numpy())
    global_longitude = abs(source_x[-1] - source_x[0] - 360) < 1e-5
    if (
        target_y[0] < source_y[0] - 1e-8
        or target_y[-1] > source_y[-1] + 1e-8
        or (
            not global_longitude
            and (target_x[0] < source_x[0] - 1e-8 or target_x[-1] > source_x[-1] + 1e-8)
        )
    ):
        msg = 'DEM does not cover the target weather cells'
        raise ValueError(msg)
    sums = np.zeros((len(lats), len(lons)), dtype=np.float64)
    areas = np.zeros_like(sums)
    chunk_y, chunk_x = field.encoding.get('chunks', (1200, 1200))
    y_blocks = _axis_blocks(field, 'latitude', chunk_y, target_y, source_y)
    x_blocks = _axis_blocks(
        field, 'longitude', chunk_x, target_x, source_x, periodic=global_longitude
    )
    for ys, rows, yy in y_blocks:
        for xs, cols, xx in x_blocks:
            data = np.nan_to_num(field.isel(latitude=ys, longitude=xs).to_numpy(), nan=0)
            sums[np.ix_(rows, cols)] += yy @ data @ xx.T
            areas[np.ix_(rows, cols)] += yy.sum(axis=1)[:, None] * xx.sum(axis=1)[None, :]
    return (sums / areas).astype(np.float32)


def _overlapping_stores(left: Store, right: Store) -> bool:
    if left is right:
        return True
    if isinstance(left, LocalStore) and isinstance(right, LocalStore):
        left_root, right_root = left.root.resolve(), right.root.resolve()
        return left_root.is_relative_to(right_root) or right_root.is_relative_to(left_root)
    if isinstance(left, MemoryStore) and isinstance(right, MemoryStore):
        return left._store_dict is right._store_dict  # noqa: SLF001
    if isinstance(left, FsspecStore) and isinstance(right, FsspecStore):
        left_fs = getattr(left.fs, 'sync_fs', left.fs)
        right_fs = getattr(right.fs, 'sync_fs', right.fs)
        left_path, right_path = left.path.strip('/'), right.path.strip('/')
        return left_fs is right_fs and (
            left_path == right_path
            or left_path.startswith(right_path + '/')
            or right_path.startswith(left_path + '/')
        )
    return False


def prepare_dem(
    source: Source,
    root: zarr.Group,
    dem_store: Store,
    output_store: Store,
    *,
    elevation: str = 'elevation',
    x: str = 'x',
    y: str = 'y',
    overwrite: bool = False,
) -> None:
    """Aggregate native elevation to the correction grid and derive its slopes."""
    if _overlapping_stores(output_store, dem_store) or _overlapping_stores(
        output_store, root.store
    ):
        msg = 'Terrain output must be separate from the source DEM and weather store'
        raise ValueError(msg)
    try:
        zarr.open_group(output_store, mode='r', use_consolidated=False)
    except FileNotFoundError:
        pass
    else:
        if not overwrite:
            msg = 'Terrain output already exists; use --overwrite to replace it'
            raise FileExistsError(msg)
    lats, lons = source.grid(root)
    with _native(dem_store, elevation, x, y) as field:
        heights = _aggregate(field, lats, lons)
    slope_we, slope_sn = _slopes(heights, lats, lons)
    ds = xr.Dataset(
        {
            name: (('latitude', 'longitude'), values, {'units': units})
            for name, values, units in [
                ('elevation', heights, 'm'),
                ('slope_we', slope_we, '1'),
                ('slope_sn', slope_sn, '1'),
            ]
        },
        coords={'latitude': lats, 'longitude': lons},
        attrs={'Conventions': 'CF-1.8', 'build_status': 'writing'},
    )
    with rioxarray.set_options(export_grid_mapping=True):
        ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        ds = ds.rio.write_crs('EPSG:4326').rio.write_coordinate_system()
    encoding = {
        name: {'chunks': (min(320, len(lats)), min(320, len(lons))), 'grid_mapping': 'spatial_ref'}
        for name in ('elevation', 'slope_we', 'slope_sn')
    }
    ds.to_zarr(output_store, mode='w', zarr_format=3, consolidated=False, encoding=encoding)
    group = zarr.open_group(output_store, mode='a', use_consolidated=False)
    group.attrs['build_status'] = 'complete'
    zarr.consolidate_metadata(output_store)


def prepare_station_terrain(
    observations: pd.DataFrame,
    dem_store: Store,
    *,
    elevation: str = 'elevation',
    x: str = 'x',
    y: str = 'y',
) -> pd.DataFrame:
    """Preserve observations and sample local native-grid elevation and slopes."""
    result = observations.copy()
    positions = result[['lat', 'lon']].drop_duplicates().copy()
    values = np.empty((len(positions), 3), dtype=np.float32)
    with _native(dem_store, elevation, x, y) as field:
        lats, lons = field.latitude.to_numpy(), field.longitude.to_numpy()
        chunk_y, chunk_x = field.encoding.get('chunks', (1200, 1200))
        ys = _chunk_slices(field, 'latitude', chunk_y)
        xs = _chunk_slices(field, 'longitude', chunk_x)
        groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for index, (lat, lon) in enumerate(positions.itertuples(index=False, name=None)):
            row, col = point_indices(lats, lons, latitude=lat, longitude=lon)
            iy = int(np.searchsorted([s.stop for s in ys], row, side='right'))
            ix = int(np.searchsorted([s.stop for s in xs], col, side='right'))
            groups.setdefault((iy, ix), []).append((index, row, col))
        for (chunk_row, chunk_col), stations in groups.items():
            start_y, start_x = max(0, ys[chunk_row].start - 1), max(0, xs[chunk_col].start - 1)
            stop_y = min(len(lats), ys[chunk_row].stop + 1)
            stop_x = min(len(lons), xs[chunk_col].stop + 1)
            heights = np.nan_to_num(
                field.isel(
                    latitude=slice(start_y, stop_y), longitude=slice(start_x, stop_x)
                ).to_numpy(),
                nan=0,
            )
            we, sn = _slopes(heights, lats[start_y:stop_y], lons[start_x:stop_x])
            for index, row, col in stations:
                pos = (row - start_y, col - start_x)
                values[index] = heights[pos], we[pos], sn[pos]
    columns = ['elevation', 'slope_we', 'slope_sn']
    positions[columns] = values
    # Join by unique coordinates without changing observation order or index.
    lookup = positions.set_index(['lat', 'lon'])
    matched = lookup.reindex(result.set_index(['lat', 'lon']).index)
    result[columns] = matched[columns].to_numpy()
    return result
