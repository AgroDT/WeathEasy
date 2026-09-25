"""Shared terrain / geography features for the bias-correction models.

Ported from the production scripts; Russia-scoped (coast reference points and the
Copernicus DEM file both cover Russia). Used by both the temperature and the
precipitation correctors, at training and inference time, so training-time station
enrichment and inference-time grid resampling read identical terrain features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr


_LOG = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    Dem = dict[str, NDArray[np.float32] | None]


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


def load_dem(path: Path) -> Dem | None:
    """Load the Copernicus DEM terrain file (elevation + slopes). None if missing."""
    if not path.is_file():
        return None
    with xr.open_dataset(path) as ds:
        return {
            'lats': ds['latitude'].to_numpy().astype(np.float32),
            'lons': ds['longitude'].to_numpy().astype(np.float32),
            'elevation': ds['elevation'].to_numpy().astype(np.float32),
            'slope_we': ds['slope_we'].to_numpy().astype(np.float32) if 'slope_we' in ds else None,
            'slope_sn': ds['slope_sn'].to_numpy().astype(np.float32) if 'slope_sn' in ds else None,
        }


def resample_dem(dem: Dem | None, lats: NDArray, lons: NDArray) -> Dem | None:
    """Linearly interpolate the DEM onto a target grid (no-op if grids already match)."""
    if dem is None:
        return None
    src_lats = dem['lats']
    src_lons = dem['lons']
    if src_lats is None or src_lons is None:
        return None

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

    out: Dem = {'lats': tgt_lats, 'lons': tgt_lons}
    for key in ('elevation', 'slope_we', 'slope_sn'):
        data = dem.get(key)
        if data is None:
            out[key] = None
            continue
        da = xr.DataArray(
            data,
            coords={'latitude': src_lats, 'longitude': src_lons},
            dims=['latitude', 'longitude'],
        )
        resampled = da.interp(latitude=tgt_lats, longitude=tgt_lons, method='linear').to_numpy()
        n_nan = int(np.isnan(resampled).sum())
        if n_nan:
            _LOG.warning(
                'DEM field "%s": %d NaN cells after resampling '
                '(missing source cells or target grid extends beyond DEM coverage) '
                '— filling with 0',
                key,
                n_nan,
            )
        out[key] = np.nan_to_num(resampled.astype(np.float32), nan=0.0)
    return out


def flat_dem(
    dem: Dem | None,
    n_pts: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Flattened elevation / slope_we / slope_sn for a (resampled) DEM; zeros if absent."""
    if dem is None:
        return np.zeros(n_pts, np.float32), np.zeros(n_pts, np.float32), np.zeros(n_pts, np.float32)
    return (
        _ravel_or_zero(dem.get('elevation'), n_pts),
        _ravel_or_zero(dem.get('slope_we'), n_pts),
        _ravel_or_zero(dem.get('slope_sn'), n_pts),
    )


def _ravel_or_zero(data: NDArray | None, n_pts: int) -> NDArray[np.float32]:
    if data is None:
        return np.zeros(n_pts, np.float32)
    return data.ravel().astype(np.float32)


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
