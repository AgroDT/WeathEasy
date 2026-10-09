"""Shared terrain / geography features for the bias-correction models.

Ported from the production scripts; Russia-scoped (coast reference points and the
Copernicus DEM dataset both cover Russia). Used by both the temperature and the
precipitation correctors, at training and inference time, so training-time station
enrichment and inference-time grid resampling read identical terrain features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from zarr.abc.store import Store


class Dem(TypedDict):
    lats: NDArray[np.float32]
    lons: NDArray[np.float32]
    elevation: NDArray[np.float32]
    slope_we: NDArray[np.float32]
    slope_sn: NDArray[np.float32]


# Reference points along Russia's sea coasts for a cheap distance-to-coast feature.
COAST_POINTS = np.array(
    [
        [59.9, 28.7],
        [60.1, 24.9],
        [56.9, 20.9],
        [54.7, 19.9],
        [69.0, 33.0],
        [70.4, 31.1],
        [71.0, 28.3],
        [72.4, 52.7],
        [73.5, 55.0],
        [76.0, 62.0],
        [77.5, 68.5],
        [72.9, 52.7],
        [66.5, 32.5],
        [65.0, 34.8],
        [64.5, 40.0],
        [67.6, 44.4],
        [73.3, 70.0],
        [73.8, 80.0],
        [74.0, 87.0],
        [76.5, 82.5],
        [73.5, 67.0],
        [73.5, 110.0],
        [73.0, 120.0],
        [72.5, 130.0],
        [74.0, 140.0],
        [70.5, 150.0],
        [70.0, 160.0],
        [68.5, 170.0],
        [66.0, 173.0],
        [64.5, 173.5],
        [63.0, 179.5],
        [59.5, 150.5],
        [56.5, 143.0],
        [53.5, 141.5],
        [51.0, 143.0],
        [52.5, 141.0],
        [46.5, 141.5],
        [48.0, 140.5],
        [44.0, 135.0],
        [43.0, 132.0],
        [47.0, 51.5],
        [44.8, 36.9],
        [41.8, 50.0],
        [37.5, 50.2],
        [46.6, 32.6],
        [44.5, 33.5],
        [43.5, 40.0],
        [43.0, 41.5],
        [46.9, 37.8],
        [47.2, 39.5],
    ],
    dtype=np.float32,
)

_BATCH = 50_000
_KM_PER_DEGREE = 111.0


def dist_to_coast(lats: NDArray, lons: NDArray) -> NDArray[np.float32]:
    """Great-circle-ish distance (km) from each point to the nearest coast reference."""
    lats = np.asarray(lats, np.float32)
    lons = np.asarray(lons, np.float32)
    out = np.empty(len(lats), np.float32)
    for s in range(0, len(lats), _BATCH):
        e = min(s + _BATCH, len(lats))
        dlat = COAST_POINTS[:, 0][None, :] - lats[s:e][:, None]
        raw_dlon = COAST_POINTS[:, 1][None, :] - lons[s:e][:, None]
        # Wrap to [-180, 180] so Chukotka (-180°) and Kamchatka (180°) stay close.
        dlon = ((raw_dlon + 180) % 360 - 180) * np.cos(np.radians(lats[s:e][:, None]))
        out[s:e] = np.sqrt(dlat**2 + dlon**2).min(axis=1) * _KM_PER_DEGREE
    return out


def point_distances_km(target: NDArray, reference: NDArray) -> NDArray[np.float64]:
    """Pairwise local distances with longitude scaled at each target latitude."""
    delta_lat = target[:, None, 0] - reference[None, :, 0]
    delta_lon = (target[:, None, 1] - reference[None, :, 1] + 180) % 360 - 180
    return 111.0 * np.hypot(
        delta_lat,
        delta_lon * np.cos(np.deg2rad(target[:, None, 0])),
    )


def load_dem(store: Store) -> Dem:
    """Load terrain features from a prepared local or S3 Zarr store."""
    with xr.open_zarr(store, consolidated=False, chunks=None, decode_coords='all') as ds:
        if ds.attrs.get('build_status', 'complete') != 'complete':
            msg = 'DEM build is incomplete; resume weatheasy dem before using it'
            raise ValueError(msg)
        dem: Dem = {
            'lats': ds['latitude'].to_numpy().astype(np.float32),
            'lons': ds['longitude'].to_numpy().astype(np.float32),
            'elevation': ds['elevation'].to_numpy().astype(np.float32),
            'slope_we': ds['slope_we'].to_numpy().astype(np.float32),
            'slope_sn': ds['slope_sn'].to_numpy().astype(np.float32),
        }
    return resample_dem(dem, dem['lats'], dem['lons'])


def resample_dem(dem: Dem, lats: NDArray, lons: NDArray) -> Dem:
    """Linearly interpolate the DEM onto a target grid (no-op if grids already match)."""
    src_lats = dem['lats']
    src_lons = dem['lons']
    for key in ('elevation', 'slope_we', 'slope_sn'):
        if not np.isfinite(dem[key]).all():
            msg = f'DEM field {key} contains missing values'
            raise ValueError(msg)

    # Use absolute tolerance only: rtol would scale with coordinate magnitude
    # (e.g. 50° * 1e-5 ≈ 0.0005° ≈ 50 m), which is too loose for grid matching.
    tgt_lats = np.asarray(lats, np.float32)
    tgt_lons = np.asarray(lons, np.float32)
    grids_match = (
        len(src_lats) == len(tgt_lats)
        and len(src_lons) == len(tgt_lons)
        and np.allclose(src_lats, tgt_lats, rtol=0, atol=1e-6)
        and np.allclose(src_lons, tgt_lons, rtol=0, atol=1e-6)
    )
    if grids_match:
        # Return with canonical target coords so the caller always gets what it asked for.
        return {**dem, 'lats': tgt_lats, 'lons': tgt_lons}

    def resample(data: NDArray[np.float32], key: str) -> NDArray[np.float32]:
        da = xr.DataArray(
            data,
            coords={'latitude': src_lats, 'longitude': src_lons},
            dims=['latitude', 'longitude'],
        )
        resampled = da.interp(latitude=tgt_lats, longitude=tgt_lons, method='linear').to_numpy()
        if not np.isfinite(resampled).all():
            msg = f'DEM field {key} is missing on the correction grid'
            raise ValueError(msg)
        return resampled.astype(np.float32)

    return {
        'lats': tgt_lats,
        'lons': tgt_lons,
        'elevation': resample(dem['elevation'], 'elevation'),
        'slope_we': resample(dem['slope_we'], 'slope_we'),
        'slope_sn': resample(dem['slope_sn'], 'slope_sn'),
    }


def flat_dem(dem: Dem) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Flattened elevation / slope_we / slope_sn for a resampled DEM."""
    return (
        dem['elevation'].ravel(),
        dem['slope_we'].ravel(),
        dem['slope_sn'].ravel(),
    )


def seasonal_field(  # noqa: PLR0917
    fields: dict[int, NDArray[np.float32]],
    grid_lats: NDArray,
    grid_lons: NDArray,
    month: int,
    lats: NDArray,
    lons: NDArray,
) -> NDArray[np.float32]:
    """Monthly field interpolated onto the target grid (no-op if grids already match)."""
    field = fields[month]
    if (
        field.shape == (len(lats), len(lons))
        and np.allclose(grid_lats, lats)
        and np.allclose(grid_lons, lons)
    ):
        return field
    da = xr.DataArray(
        field,
        coords={'latitude': grid_lats, 'longitude': grid_lons},
        dims=['latitude', 'longitude'],
    )
    return da.interp(latitude=lats, longitude=lons, method='linear').to_numpy().astype(np.float32)
