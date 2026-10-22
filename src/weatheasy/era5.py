"""Orchestrate ERA5-Land cached inputs, daily reductions and restartable publication."""

from __future__ import annotations

import calendar
import logging
import time
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np

from weatheasy import const
from weatheasy.era5_daily import process_day, validate_boundary
from weatheasy.era5_download import CdsRequests
from weatheasy.era5_grib import merge_grib_fields, read_grib
from weatheasy.era5_requests import Part, boundary_part, month_parts, needed_files, next_month
from weatheasy.era5_stage import DailyStage, month_marker, publish_stage, resume_stage
from weatheasy.raw_cache import DownloadCache, download_cache
from weatheasy.storage import (
    StorageError,
    group_attrs,
    group_exists,
    last_date,
    open_dataset,
    set_group_attrs,
)
from weatheasy.util import utc_now


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import xarray as xr
    from zarr.abc.store import Store

    from weatheasy.file_storage import FileStore


LOGGER = logging.getLogger(__name__)


def available_period(store: Store) -> tuple[date, date]:
    """Return the committed time range, including incomplete years.

    last_month is a convenience marker. A crash after committing data but before
    updating that marker must neither hide data nor cause a duplicate append.
    """
    with open_dataset(store, 'daily') as ds:
        times = ds.time.to_numpy().astype('datetime64[D]')
        first = date.fromisoformat(str(times[0]))
        last = date.fromisoformat(str(times[-1]))
    return first, last


def download_era5_data(
    store: Store,
    download_dir: str | Path | FileStore | None = None,
    *,
    work_dir: str | Path | None = None,
    work_max_age_days: int = 7,
    bbox: tuple[float, float, float, float] | None = None,
    max_pending_requests: int = 3,
    proxy: str | None = None,
) -> None:
    """Append complete months through the last month with available midnight data.

    bbox is (north, west, south, east), without antimeridian crossing. The CDS
    publication lag is five days; only whole months are requested. Run as an
    external job. No download is triggered by API or prepared-data reads.
    """
    if max_pending_requests < 1:
        msg = 'max_pending_requests must be at least 1'
        raise ValueError(msg)
    bbox = bbox or (90.0, -180.0, -90.0, 180.0)
    north, west, south, east = bbox
    if not np.isfinite(bbox).all() or not (
        -90 <= south < north <= 90 and -180 <= west < east <= 180
    ):
        msg = 'Invalid ERA5 bbox (north, west, south, east)'
        raise StorageError(msg)
    if (
        group_exists(store, '')
        and (stored_bbox := group_attrs(store, '').get('request_bbox')) is not None
        and stored_bbox != list(bbox)
    ):
        msg = 'Existing ERA5 store has a different requested bbox'
        raise StorageError(msg)
    set_group_attrs(store, '', request_bbox=list(bbox))
    path = 'daily'
    previous = last_date(store, path)
    latest = utc_now().date() - 5 * const.ONE_DAY
    if previous:
        marker = month_marker(previous)
        if group_attrs(store, '').get('last_month') != marker:
            # Recover a crash after committing daily data but before journal cleanup.
            set_group_attrs(store, '', last_month=marker)
    stage, first, count = resume_stage(store, bbox, work_dir, previous)
    with download_cache(download_dir, 'era5', work_dir=work_dir, target=str(store)) as cache:
        needed = needed_files(first.replace(day=1), latest, bbox)
        cache.cleanup(needed.__contains__, max_age_days=work_max_age_days)
        if next_month(first) > latest:
            publish_stage(stage, store)
            set_group_attrs(store, '', era5_requests={})
            return
        try:
            # The CDS client is optional and needed only for a pending download.
            import cdsapi  # noqa: PLC0415
            import requests  # noqa: PLC0415
        except ImportError as err:
            msg = 'ERA5 preparation requires weatheasy[prepare]'
            raise ImportError(msg) from err
        with requests.Session() as session:
            if proxy is not None:
                # An explicit proxy takes precedence over environment proxy settings.
                session.trust_env = False
                session.proxies.update({'http': proxy, 'https': proxy})
            # Keep cdsapi configuration/logging while using its DSS backend's job API.
            client = cast('Any', cdsapi.Client(wait_until_complete=False, session=session)).client
            jobs = CdsRequests(client, store, max_pending_requests)
            inputs = _MonthlyInputs(jobs, cache, bbox)
            _prepare_months(inputs, stage, store, first, latest, count=count)
        publish_stage(stage, store)


def _prepare_months(
    inputs: _MonthlyInputs,
    stage: DailyStage,
    store: Store,
    first: date,
    latest: date,
    *,
    count: int,
) -> None:
    month = first.replace(day=1)
    capacity = 128 - count % 128
    while next_month(month) <= latest:
        started = time.monotonic()
        inputs.plan(month, latest)
        LOGGER.info('Preparing ERA5-Land %s; local staging in %s', month, stage.directory)
        main, boundary = inputs.retrieve_month(month, latest)
        try:
            validate_boundary(main, boundary)
            for index, day in enumerate(
                _daily_batches(main, boundary, month, inputs.bbox, begin=first)
            ):
                if index == 0:
                    stage.check_compatible(day)
                stage.append(day)
                if stage.written == capacity:
                    publish_stage(stage, store)
                    capacity = 128
        finally:
            main.close()
            boundary.close()
        # Daily reductions are checkpointed locally, so hourly files can be released.
        inputs.release(month)
        LOGGER.info('Staged ERA5-Land %s in %.1f s', month, time.monotonic() - started)
        month = next_month(month)
        first = month


class _MonthlyInputs:
    """Reuse cached GRIB parts and keep next-month inputs until their reductions finish."""

    def __init__(self, jobs: CdsRequests, cache: DownloadCache, bbox: tuple) -> None:
        self.jobs = jobs
        self.cache = cache
        self.bbox = bbox
        self.parts: dict[str, Part] = {}

    def plan(self, month: date, latest: date) -> None:
        # Prioritize this month's inputs, including the following midnights,
        # before prefetching inputs for the next month.
        for owner in (month, next_month(month)):
            if next_month(owner) <= latest:
                for part in [
                    *month_parts(owner, self.bbox),
                    boundary_part(owner, latest, self.bbox),
                ]:
                    self.parts[part.key] = part
        self.jobs.retain(self.parts)
        self._fill()

    def _fill(self) -> None:
        self.jobs.fill(part for part in self.parts.values() if not self.cache.exists(part.filename))

    def retrieve_month(self, month: date, latest: date) -> tuple[xr.Dataset, xr.Dataset]:
        main = month_parts(month, self.bbox)
        pieces = self._retrieve([*main, boundary_part(month, latest, self.bbox)])
        return merge_grib_fields(*pieces[: len(main)]), pieces[-1]

    def retrieve(self, month: date, *, boundary: bool) -> xr.Dataset:
        return merge_grib_fields(*self._retrieve(month_parts(month, self.bbox, boundary=boundary)))

    def _retrieve(self, parts: list[Part]) -> list[xr.Dataset]:
        self.parts.update((part.key, part) for part in parts)
        self._fill()
        wanted = {part.key for part in parts if not self.cache.exists(part.filename)}
        pieces = {part.key: self._read(part) for part in parts if part.key not in wanted}
        while wanted:
            key = self.jobs.ready(wanted)
            part = self.parts[key]
            self.jobs.download(key, self.cache.file(part.filename))
            wanted.remove(key)
            # Let CDS work while this file is uploaded and processed.
            self._fill()
            pieces[key] = self._read(part)
        return [pieces[part.key] for part in parts]

    def release(self, month: date) -> None:
        # Retain full next-month midnight files; release a one-midnight boundary
        # together with the month that consumed it.
        parts = [
            part
            for part in self.parts.values()
            if part.month == month or (part.boundary and part.month == next_month(month))
        ]
        for part in parts:
            self.parts.pop(part.key)
        self.jobs.release(part.key for part in parts)
        self.cache.release(self.cache.root / part.filename for part in parts)

    def _read(self, part: Part) -> xr.Dataset:
        path = self.cache.file(part.filename)
        # Also retry a publication that failed after a complete local download.
        started = time.monotonic()
        self.cache.publish([path])
        LOGGER.info('Cached %s in %.1f s', path.name, time.monotonic() - started)
        return read_grib(path, part.required)


def _daily_batches(
    main: xr.Dataset, boundary: xr.Dataset, month: date, bbox: tuple, *, begin: date | None = None
):
    """Reduce only days not already committed or checkpointed locally."""
    days = calendar.monthrange(month.year, month.month)[1]
    for offset in range((begin - month).days if begin else 0, days):
        day = month + offset * const.ONE_DAY
        LOGGER.info('Reducing and staging ERA5-Land %s (%s/%s)', day, offset + 1, days)
        ds = process_day(main, boundary, day)
        ds.attrs['request_bbox'] = list(bbox)
        yield ds
        del ds
