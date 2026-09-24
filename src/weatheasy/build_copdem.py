"""Build a Copernicus DEM (GLO-90) terrain file for the bias-correction models.

Downloads Copernicus DEM GLO-90 tiles (1°x1° COGs) anonymously from the AWS Open
Data bucket, aggregates them to a coarse regular grid (default 0.1°) and writes a
NetCDF with ``elevation``, ``slope_we`` and ``slope_sn`` — the same schema the
correctors previously read from the FABDEM file, so it is a drop-in replacement.

This is a one-off build step (terrain is static); it is NOT part of the realtime
download path.

Example (full Russia, 0.1°):
    python -m weatheasy.build_copdem --area 82 19 41 180 -o copdem_russia_01deg.nc

Smoke test (single tile near Moscow):
    python -m weatheasy.build_copdem --area 56 37 55 38 -o copdem_moscow.nc
"""

from __future__ import annotations

import logging
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from itertools import product
from pathlib import Path

import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.enums import Resampling
from requests import Session
from requests.adapters import HTTPAdapter


_LOG = logging.getLogger('build_copdem')

_HOST = 'copernicus-dem-90m.s3.amazonaws.com'
_RES_CODE = 30  # 10 = GLO-30, 30 = GLO-90
_M_PER_DEGREE = 111_320.0  # metres per degree of latitude (approx.)


def _session() -> Session:
    s = Session()
    s.mount('https://', HTTPAdapter(max_retries=3))
    return s


def _tile_name(tlat: int, tlon: int) -> str:
    ns = 'N' if tlat >= 0 else 'S'
    ew = 'E' if tlon >= 0 else 'W'
    return f'Copernicus_DSM_COG_{_RES_CODE}_{ns}{abs(tlat):02d}_00_{ew}{abs(tlon):03d}_00_DEM'


def _fetch_tile(
    tlat: int,
    tlon: int,
    cache_dir: Path,
    session: Session,
) -> Path | None:
    """Download one 1°x1° DEM tile to the cache. Returns None for missing (ocean) tiles."""
    name = _tile_name(tlat, tlon)
    path = cache_dir / f'{name}.tif'
    if path.is_file():
        return path
    url = f'https://{_HOST}/{name}/{name}.tif'
    _LOG.info('Downloading %s', url)
    with session.get(url, timeout=180) as response:
        if response.status_code == 404:
            _LOG.info('  no tile (ocean / out of coverage): %s', name)
            return None
        if not response.ok:
            msg = f'Failed to download {url}: HTTP {response.status_code}'
            raise RuntimeError(msg)
        path.write_bytes(response.content)
    return path


def _read_tile_block(path: Path, cells_per_degree: int) -> np.ndarray:
    """Average-resample a tile to a ``cells_per_degree`` x ``cells_per_degree`` block.

    Output is oriented south-to-north.
    """
    with rasterio.open(path) as ds:
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


def build_copdem(
    area: tuple[int, int, int, int],
    resolution: float,
    output: Path,
    cache_dir: Path,
) -> None:
    north, west, south, east = area
    if not (south < north and west < east):
        msg = f'Invalid --area {area}; expected N W S E with N>S and E>W'
        raise ValueError(msg)

    cells_per_degree = round(1.0 / resolution)
    if not math.isclose(cells_per_degree * resolution, 1.0):
        msg = f'--resolution {resolution} must divide 1.0 (e.g. 0.1, 0.05)'
        raise ValueError(msg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    n_lat = round((north - south) / resolution)
    n_lon = round((east - west) / resolution)
    lats = (south + (np.arange(n_lat) + 0.5) * resolution).astype(np.float32)
    lons = (west + (np.arange(n_lon) + 0.5) * resolution).astype(np.float32)
    elev = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    session = _session()
    tiles = list(product(range(south, north), range(west, east)))
    n_tiles = len(tiles)
    found = 0
    for done, (tlat, tlon) in enumerate(tiles, start=1):
        if done % 50 == 0 or done == n_tiles:
            _LOG.info('Tiles: %d/%d (found %d)', done, n_tiles, found)

        path = _fetch_tile(tlat, tlon, cache_dir, session)
        if path is None:
            continue

        block = _read_tile_block(path, cells_per_degree)
        r0 = (tlat - south) * cells_per_degree
        c0 = (tlon - west) * cells_per_degree
        elev[r0 : r0 + cells_per_degree, c0 : c0 + cells_per_degree] = block
        found += 1

    if found == 0:
        msg = 'No tiles downloaded — check --area covers land within Copernicus DEM coverage'
        raise RuntimeError(msg)

    nan_cells = int(np.isnan(elev).sum())
    if nan_cells:
        _LOG.warning(
            '%d grid cells remain NaN after tile collection (%.1f%% of grid) — '
            'treating as ocean/no-coverage and filling with 0',
            nan_cells,
            100.0 * nan_cells / elev.size,
        )
    elev = np.nan_to_num(elev, nan=0.0)
    slope_we, slope_sn = _slopes(elev, lats, resolution)
    _write_netcdf(output, lats, lons, elev, slope_we, slope_sn)
    _LOG.info(
        'Wrote %s  grid=%dx%d  tiles=%d  elev[min=%.0f max=%.0f]',
        output,
        n_lat,
        n_lon,
        found,
        float(elev.min()),
        float(elev.max()),
    )


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


def _write_netcdf(
    output: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    elev: np.ndarray,
    slope_we: np.ndarray,
    slope_sn: np.ndarray,
) -> None:
    with nc.Dataset(output, 'w') as ds:
        ds.createDimension('latitude', len(lats))
        ds.createDimension('longitude', len(lons))
        for name, dims, data in (
            ('latitude', ('latitude',), lats),
            ('longitude', ('longitude',), lons),
            ('elevation', ('latitude', 'longitude'), elev),
            ('slope_we', ('latitude', 'longitude'), slope_we),
            ('slope_sn', ('latitude', 'longitude'), slope_sn),
        ):
            var = ds.createVariable(name, 'f4', dims, zlib=True, complevel=4)
            var[:] = data
        ds.source = f'Copernicus DEM GLO-90 ({_HOST})'
        ds.title = 'Terrain features for bias correction (elevation, slope_we, slope_sn)'


def main() -> None:
    logging.basicConfig(
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        level=logging.INFO,
    )
    parser = ArgumentParser('build_copdem', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--area',
        type=int,
        nargs=4,
        metavar=('N', 'W', 'S', 'E'),
        default=[82, 19, 41, 180],
        help='bounding box in whole degrees (Russia by default)',
    )
    parser.add_argument('--resolution', type=float, default=0.1, help='output grid step in degrees')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('copdem_russia_01deg.nc'),
        help='output NetCDF path',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('copdem-cache'),
        help='directory for downloaded tiles (enables resume)',
    )
    args = parser.parse_args()
    build_copdem(tuple(args.area), args.resolution, args.output, args.cache_dir)


if __name__ == '__main__':
    main()
