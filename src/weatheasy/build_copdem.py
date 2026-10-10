"""Build a Copernicus DEM (GLO-90) terrain dataset for the bias-correction models.

Reads Copernicus DEM GLO-90 tiles (1°x1° COGs) directly from the AWS Open
Data bucket over HTTP, aggregates them to a coarse regular grid (default 0.1°) and writes a
spatial Zarr v3 dataset with ``elevation``, ``slope_we`` and ``slope_sn``.
The resumable published store can be local or in S3; source tiles are not cached.

This is a one-off build step (terrain is static); it is NOT part of the realtime
download path.

Example (full Russia, 0.1°):
    weatheasy dem --bbox 82 19 41 180 -o copdem_russia_01deg.zarr

Smoke test (single tile near Moscow):
    weatheasy dem --bbox 56 37 55 38 -o copdem_moscow.zarr
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import rasterio
import rioxarray
import xarray as xr
import zarr
from rasterio.enums import Resampling
from rasterio.session import DummySession
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


if TYPE_CHECKING:
    from zarr.abc.store import Store


_LOG = logging.getLogger('build_copdem')

_HOST = 'copernicus-dem-90m.s3.amazonaws.com'
_RES_CODE = 30  # 10 = GLO-30, 30 = GLO-90
_CHUNK_TILES = 32  # A full block covers 32 x 32 source tiles at any output resolution.
_M_PER_DEGREE = 111_320.0  # metres per degree of latitude (approx.)


def _tile_name(tlat: int, tlon: int) -> str:
    ns = 'N' if tlat >= 0 else 'S'
    ew = 'E' if tlon >= 0 else 'W'
    return f'Copernicus_DSM_COG_{_RES_CODE}_{ns}{abs(tlat):02d}_00_{ew}{abs(tlon):03d}_00_DEM'


def _read_tile_block(url: str, cells_per_degree: int) -> np.ndarray | None:
    """Average-resample a tile to a ``cells_per_degree`` x ``cells_per_degree`` block.

    Output is oriented south-to-north.
    """
    path = f'/vsicurl/{url}' if url.startswith(('http://', 'https://')) else url
    try:
        source = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        if 'HTTP response code: 404' in str(exc):
            return None
        raise
    with source as ds:
        block = ds.read(
            1,
            out_shape=(cells_per_degree, cells_per_degree),
            resampling=Resampling.average,
        )
        nodata = ds.nodata

    block = block.astype(np.float32)
    if nodata is not None:
        block[block == nodata] = np.nan

    # rasterio rows go north -> south; our output grid goes south -> north.
    return block[::-1, :]


class BuildConfig(TypedDict):
    bbox: list[int]
    resolution: float
    source: str
    algorithm_version: int
    chunk_tiles: list[int]


def _build_group(
    output: Store, config: BuildConfig, lats: np.ndarray, lons: np.ndarray
) -> zarr.Group:
    group = zarr.open_group(output, mode='a', zarr_format=3, use_consolidated=False)
    previous = group.attrs.get('build_config')
    if previous is None:
        if list(group.keys()) or dict(group.attrs):
            msg = 'Target Zarr is not a resumable DEM; choose a new output path'
            raise ValueError(msg)
        group.attrs.update(
            build_config={
                'bbox': config['bbox'],
                'resolution': config['resolution'],
                'source': config['source'],
                'algorithm_version': config['algorithm_version'],
                'chunk_tiles': config['chunk_tiles'],
            },
            build_status='initializing',
        )
    elif previous != config:
        msg = 'DEM build parameters differ from the target Zarr; choose a new output path'
        raise ValueError(msg)
    if group.attrs['build_status'] == 'initializing':
        for key in list(group.keys()):
            del group[key]
        _initialize_zarr(output, lats, lons, config)
        group = zarr.open_group(output, mode='a')
        group.attrs['build_status'] = 'building'
    return group


def build_copdem(
    bbox: tuple[int, int, int, int],
    resolution: float,
    output: Store,
) -> None:
    north, west, south, east = bbox
    if not (south < north and west < east):
        msg = f'Invalid --bbox {bbox}; expected N W S E with N>S and E>W'
        raise ValueError(msg)

    if not math.isfinite(resolution) or resolution <= 0:
        msg = '--resolution must be finite and positive'
        raise ValueError(msg)
    cells_per_degree = round(1.0 / resolution)
    if not math.isclose(cells_per_degree * resolution, 1.0):
        msg = f'--resolution {resolution} must divide 1.0 (e.g. 0.1, 0.05)'
        raise ValueError(msg)

    n_lat = round((north - south) / resolution)
    n_lon = round((east - west) / resolution)
    lats = (south + (np.arange(n_lat) + 0.5) * resolution).astype(np.float32)
    lons = (west + (np.arange(n_lon) + 0.5) * resolution).astype(np.float32)

    config: BuildConfig = {
        'bbox': list(bbox),
        'resolution': resolution,
        'source': _HOST,
        'algorithm_version': 2,
        'chunk_tiles': [
            min(north - south, _CHUNK_TILES),
            min(east - west, _CHUNK_TILES),
        ],
    }
    group = _build_group(output, config, lats, lons)
    if group.attrs['build_status'] == 'complete':
        zarr.consolidate_metadata(output)
        _LOG.info('DEM is already complete: %s', output)
        return

    status = cast('zarr.Array', group['tile_status'])
    elevation = cast('zarr.Array', group['elevation'])
    with (
        logging_redirect_tqdm(),
        rasterio.Env(
            session=DummySession(),
            GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
            GDAL_HTTP_MAX_RETRY='3',
            GDAL_HTTP_RETRY_DELAY='1',
        ),
    ):
        _write_elevation_chunks(elevation, status, config)

    if not np.any(status[:] == 1):
        msg = 'No tiles found — check --bbox covers land within Copernicus DEM coverage'
        raise RuntimeError(msg)
    group.attrs['build_status'] = 'computing_slopes'
    _LOG.info('Computing terrain slopes')
    elev = np.asarray(elevation[:])
    slope_we, slope_sn = _slopes(elev, lats, resolution)
    cast('zarr.Array', group['slope_we'])[:] = slope_we
    cast('zarr.Array', group['slope_sn'])[:] = slope_sn
    group.attrs['build_status'] = 'complete'
    zarr.consolidate_metadata(output)
    _LOG.info('Completed %s  grid=%dx%d', output, n_lat, n_lon)


def _read_chunk(
    south: int, west: int, tile_rows: int, tile_cols: int, cells: int
) -> tuple[np.ndarray, np.ndarray]:
    elevation = np.zeros((tile_rows * cells, tile_cols * cells), dtype=np.float32)
    statuses = np.ones((tile_rows, tile_cols), dtype=np.uint8)
    with tqdm(
        total=tile_rows * tile_cols,
        desc='Tiles',
        unit='tile',
        position=1,
        leave=False,
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for row in range(tile_rows):
            for col in range(tile_cols):
                name = _tile_name(south + row, west + col)
                url = f'https://{_HOST}/{name}/{name}.tif'
                progress.set_postfix_str(name, refresh=True)
                block = _read_tile_block(url, cells)
                if block is None:
                    statuses[row, col] = 2
                else:
                    elevation[row * cells : (row + 1) * cells, col * cells : (col + 1) * cells] = (
                        np.nan_to_num(block, nan=0.0)
                    )
                progress.update()
    return elevation, statuses


def _write_elevation_chunks(elevation: zarr.Array, status: zarr.Array, config: BuildConfig) -> None:
    north, west, south, east = config['bbox']
    chunk_rows, chunk_cols = config['chunk_tiles']
    cells = round(1 / config['resolution'])
    saved_status = np.asarray(status[:])
    pending = []
    total = 0
    for row in range(0, north - south, chunk_rows):
        for col in range(0, east - west, chunk_cols):
            rows = min(chunk_rows, north - south - row)
            cols = min(chunk_cols, east - west - col)
            total += 1
            if np.any(saved_status[row : row + rows, col : col + cols] == 0):
                pending.append((row, col, rows, cols))
    with tqdm(
        total=total,
        initial=total - len(pending),
        desc='Chunks',
        unit='chunk',
        position=0,
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for row, col, rows, cols in pending:
            block, statuses = _read_chunk(south + row, west + col, rows, cols, cells)
            region = (
                slice(row * cells, (row + rows) * cells),
                slice(col * cells, (col + cols) * cells),
            )
            # Each region covers one output chunk; commit progress after its data.
            elevation[region] = block
            status[row : row + rows, col : col + cols] = statuses
            progress.update()


def _slopes(
    elev: np.ndarray,
    lats: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless terrain slope (m/m) west-east and south-north on the output grid."""
    dlat_m = resolution * _M_PER_DEGREE
    dlon_m = resolution * _M_PER_DEGREE * np.cos(np.radians(lats))
    slope_sn = (np.gradient(elev, axis=0) / dlat_m).astype(np.float32)
    slope_we = (np.gradient(elev, axis=1) / dlon_m[:, None]).astype(np.float32)

    return np.nan_to_num(slope_we), np.nan_to_num(slope_sn)


def _initialize_zarr(
    output: Store,
    lats: np.ndarray,
    lons: np.ndarray,
    config: BuildConfig,
) -> None:
    """Write coordinates and create field metadata without allocating data chunks."""
    ds = xr.Dataset(
        coords={'latitude': lats, 'longitude': lons},
        attrs={
            'build_config': config,
            'build_status': 'initializing',
            'source': f'Copernicus DEM GLO-90 ({_HOST})',
            'title': 'Terrain features for bias correction (elevation, slope_we, slope_sn)',
            'Conventions': 'CF-1.8',
        },
    )
    ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
    with rioxarray.set_options(export_grid_mapping=True):
        ds = ds.rio.write_crs('EPSG:4326').rio.write_coordinate_system().rio.write_transform()
    # Only the coordinate vectors and scalar spatial_ref have actual data here.
    ds.to_zarr(output, mode='a', zarr_format=3, consolidated=False)
    group = zarr.open_group(output, mode='a')
    cells = round(1 / config['resolution'])
    chunk_rows, chunk_cols = config['chunk_tiles']
    for name, units, long_name in (
        ('elevation', 'm', 'Elevation'),
        ('slope_we', '1', 'West-east terrain slope'),
        ('slope_sn', '1', 'South-north terrain slope'),
    ):
        group.create_array(
            name,
            shape=(len(lats), len(lons)),
            chunks=(chunk_rows * cells, chunk_cols * cells),
            dtype='float32',
            fill_value=np.nan,
            dimension_names=('latitude', 'longitude'),
            attributes={'units': units, 'long_name': long_name, 'grid_mapping': 'spatial_ref'},
        )
    north, west, south, east = config['bbox']
    group.create_array(
        'tile_status',
        shape=(north - south, east - west),
        chunks=(chunk_rows, chunk_cols),
        dtype='uint8',
        fill_value=0,
        dimension_names=('tile_latitude', 'tile_longitude'),
        attributes={'codes': '0=pending, 1=done, 2=missing'},
    )
