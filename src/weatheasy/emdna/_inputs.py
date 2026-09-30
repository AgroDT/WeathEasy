"""File inputs at the EMDNA command boundary."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from fsspec.implementations.local import LocalFileSystem


if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import date

    from ._base import EMDNABase
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)


def read_stations(station_file: FileStore) -> pd.DataFrame:
    with station_file.open() as stream:
        return pd.read_csv(stream)


@contextmanager
def monthly_era5(directory: FileStore | None, month: str) -> Iterator[xr.Dataset | None]:
    """Open one monthly input, staging remote NetCDF on the job's local disk."""
    if directory is None:
        yield None
        return
    with ExitStack() as stack:
        try:
            files = sorted(directory.glob(f'*{month.replace("-", "")}*daily.nc'))
            if not files:
                _LOG.warning('No daily ERA5 file for %s; using stations', month)
                dataset = None
            else:
                path = files[0]
                if not isinstance(directory.fs, LocalFileSystem):
                    temporary = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix='weatheasy-emdna-')
                    )
                    local = Path(temporary) / 'daily.nc'
                    with directory.fs.open(path, 'rb') as source, local.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    path = str(local)
                dataset = stack.enter_context(xr.open_dataset(path))
        except Exception as exc:
            # Preserve the existing policy: inaccessible/broken ERA5 falls back to stations.
            _LOG.warning('ERA5 load failed for %s: %s', month, exc)
            dataset = None
        yield dataset


def process_period(
    processor: EMDNABase,
    era5_dir: FileStore | None,
    start: date | None = None,
    end: date | None = None,
) -> list[dict]:
    """Bind one ERA5 month at a time without concatenating the full input collection."""
    selected = pd.Series(pd.to_datetime(processor.pending_dates(start, end)))
    results = []
    for month, observations in selected.groupby(selected.dt.to_period('M'), sort=True):
        with monthly_era5(era5_dir, str(month)) as dataset:
            processor.era5 = dataset
            try:
                batch = processor.process_period(
                    observations.min().date(), observations.max().date()
                )
            finally:
                processor.era5 = None
        results.extend(batch)
        if any(result['status'] not in ('success', 'exists') for result in batch):
            break
    return results
