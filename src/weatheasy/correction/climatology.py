"""Frozen cell/month climatology shared by correction training and inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fsspec.implementations.local import LocalFileSystem

from weatheasy.storage import open_dataset


if TYPE_CHECKING:
    import zarr
    from numpy.typing import NDArray

    from .artifacts import Artifacts
    from .sources import Source


SCHEMA = 'cell_month_v1'
TEMP_FILES = ('climatology.npz', 'model.joblib', 'seasonal_bias_grid.npz', 'lats.npy', 'lons.npy')
PRECIP_FILES = (
    'climatology.npz',
    'clf.joblib',
    'reg.joblib',
    'fallback_clf.joblib',
    'fallback_reg.joblib',
    'train_coords.npy',
    'lats.npy',
    'lons.npy',
    'seasonal.npz',
)


def _digest(directory: Artifacts, name: str) -> str:
    digest = hashlib.sha256()
    with directory.open(name) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Artifacts, files: tuple[str, ...]) -> None:
    """Publish hashes last, so interrupted retraining cannot mix old and new files."""
    manifest = {
        'schema': SCHEMA,
        'files': {name: _digest(directory, name) for name in files},
    }
    if isinstance(directory.fs, LocalFileSystem):
        # fsspec local transactions stage in /tmp, which may be another filesystem.
        # Keep replacement atomic by staging alongside the published manifest.
        with directory.open('manifest.json.tmp', 'wb') as stream:
            stream.write(json.dumps(manifest, sort_keys=True).encode('utf-8'))
        (Path(directory.root) / 'manifest.json.tmp').replace(Path(directory.root) / 'manifest.json')
    else:
        with directory.fs.transaction, directory.open('manifest.json', 'wb') as stream:
            stream.write(json.dumps(manifest, sort_keys=True).encode('utf-8'))


def verify_manifest(directory: Artifacts, files: tuple[str, ...]) -> None:
    if not directory.exists('manifest.json'):
        msg = f'{directory.root}: old or incomplete correction artifacts; retrain'
        raise ValueError(msg)
    with directory.open('manifest.json') as stream:
        manifest = json.load(stream)
    if manifest.get('schema') != SCHEMA or set(manifest.get('files', {})) != set(files):
        msg = f'{directory.root}: incompatible correction artifacts; retrain'
        raise ValueError(msg)
    if any(
        not directory.exists(name) or _digest(directory, name) != manifest['files'][name]
        for name in files
    ):
        msg = f'{directory.root}: incomplete or modified correction artifacts; retrain'
        raise ValueError(msg)


@dataclass(frozen=True)
class Climatology:
    """Twelve calendar-month fields on the source's exact correction grid."""

    mean: NDArray[np.float32]
    wet_fraction: NDArray[np.float32] | None
    lats: NDArray[np.float32]
    lons: NDArray[np.float32]
    metadata: dict[str, str | int | float]

    def save(self, directory: Artifacts) -> None:
        values: dict[str, Any] = {
            'mean': self.mean,
            'lats': self.lats,
            'lons': self.lons,
            'metadata': json.dumps(self.metadata, sort_keys=True),
        }
        if self.wet_fraction is not None:
            values['wet_fraction'] = self.wet_fraction
        with directory.open('climatology.npz', 'wb') as stream:
            np.savez_compressed(stream, **values)

    @classmethod
    def load(cls, directory: Artifacts, source: Source, variable: str) -> Climatology:
        if not directory.exists('climatology.npz'):
            msg = f'{directory.root}: old correction artifacts have no cell/month climatology; '
            msg += 'retrain'
            raise ValueError(msg)
        with (
            directory.open('climatology.npz') as stream,
            np.load(stream, allow_pickle=False) as data,
        ):
            metadata = json.loads(str(data['metadata']))
            if (
                metadata.get('schema') != SCHEMA
                or metadata.get('source') != source.name
                or metadata.get('variable') != variable
                or metadata.get('group_path') != source.group_path
                or metadata.get('scale')
                != (source.precip_to_mm if variable == source.precip_var else 1.0)
                or not isinstance(metadata.get('training_id'), str)
                or not isinstance(metadata.get('days_sha256'), str)
            ):
                msg = f'{directory.root}: incompatible correction artifacts; retrain'
                raise ValueError(msg)
            mean = data['mean'].astype(np.float32)
            lats = data['lats'].astype(np.float32)
            lons = data['lons'].astype(np.float32)
            wet = data['wet_fraction'].astype(np.float32) if 'wet_fraction' in data else None
        shape = (12, len(lats), len(lons))
        if mean.shape != shape or (wet is not None and wet.shape != shape):
            msg = f'{directory.root}: invalid cell/month climatology shape'
            raise ValueError(msg)
        if variable == source.precip_var and wet is None:
            msg = f'{directory.root}: missing precipitation wet fraction; retrain'
            raise ValueError(msg)
        return cls(mean, wet, lats, lons, metadata)

    def require_grid(self, lats: NDArray, lons: NDArray) -> None:
        if not (np.array_equal(self.lats, lats) and np.array_equal(self.lons, lons)):
            msg = 'Correction climatology grid differs from prepared reanalysis; retrain'
            raise ValueError(msg)

    def sample(self, month: NDArray, lat_index: NDArray, lon_index: NDArray) -> NDArray:
        return self.mean[month - 1, lat_index, lon_index]

    def require_wet_fraction(self) -> NDArray[np.float32]:
        if self.wet_fraction is None:
            msg = 'Precipitation climatology has no wet fraction; retrain'
            raise ValueError(msg)
        return self.wet_fraction


def build_climatology(
    source: Source,
    root: zarr.Group,
    variable: str,
    scale: float,
    *,
    wet_threshold: float | None = None,
) -> Climatology:
    """Stream committed raw days, excluding missing cells from every denominator."""
    with open_dataset(root.store, source.path(root)) as ds:
        view = source._crop(ds)  # noqa: SLF001
        lats = view.latitude.to_numpy().astype(np.float32)
        lons = view.longitude.to_numpy().astype(np.float32)
        days = view.time.to_numpy().astype('datetime64[D]')
        shape = (12, len(lats), len(lons))
        sums = np.zeros(shape, np.float64)
        counts = np.zeros(shape, np.int32)
        wets = np.zeros(shape, np.int32) if wet_threshold is not None else None
        for i, day in enumerate(days):
            month = int(str(day)[5:7]) - 1
            field = view[variable].isel(time=i).to_numpy().astype(np.float64) * scale
            if wets is not None:
                field = np.maximum(field, 0)
            finite = np.isfinite(field)
            sums[month] += np.where(finite, field, 0)
            counts[month] += finite
            if wets is not None and wet_threshold is not None:
                wets[month] += finite & (field >= wet_threshold)
    mean = np.full(shape, np.nan, np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    wet_fraction = None
    if wets is not None:
        wet_fraction = np.full(shape, np.nan, np.float32)
        np.divide(wets, counts, out=wet_fraction, where=counts > 0)
    metadata: dict[str, str | int | float] = {
        'schema': SCHEMA,
        'source': source.name,
        'group_path': source.group_path,
        'variable': variable,
        'scale': scale,
        'first_day': str(days[0]),
        'last_day': str(days[-1]),
        'day_count': len(days),
        'days_sha256': hashlib.sha256(days.astype('int64').tobytes()).hexdigest(),
    }
    return Climatology(mean, wet_fraction, lats, lons, metadata)
