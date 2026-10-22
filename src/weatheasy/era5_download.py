"""Bounded, restartable CDS job queue and atomic downloads to caller-owned paths."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from weatheasy.storage import group_attrs, set_group_attrs


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pathlib import Path

    from zarr.abc.store import Store

    from weatheasy.era5_requests import Part


LOGGER = logging.getLogger(__name__)


class CdsRequests:
    """Own CDS jobs and their journal; callers own cache policy and GRIB processing."""

    def __init__(self, client: Any, store: Store, limit: int = 3) -> None:
        self.client = client
        self.store = store
        self.limit = limit
        self.pending: dict[str, Any] = {}
        self.requests: dict[str, str] = group_attrs(store, '').get('era5_requests', {})

    def retain(self, keys: Collection[str]) -> None:
        self.requests = {key: value for key, value in self.requests.items() if key in keys}
        self._save()

    def release(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.requests.pop(key, None)
        self._save()

    def _save(self) -> None:
        set_group_attrs(self.store, '', era5_requests=self.requests)

    def fill(self, parts: Iterable[Part]) -> None:
        for part in parts:
            if len(self.pending) >= self.limit:
                break
            if part.key not in self.pending:
                self.pending[part.key] = self._request(part)

    def _request(self, part: Part) -> Any:
        from requests import HTTPError  # noqa: PLC0415

        if request_id := self.requests.get(part.key):
            try:
                remote = self.client.get_remote(request_id)
            except HTTPError as exc:
                if exc.response is None or exc.response.status_code not in (404, 410):
                    raise
                LOGGER.info('ERA5-Land request %s is no longer available; resubmitting', request_id)
            else:
                if remote.status not in ('failed', 'rejected', 'dismissed', 'deleted'):
                    LOGGER.info('Resuming ERA5-Land request %s', request_id)
                    return remote
                LOGGER.info('ERA5-Land request %s has failed or expired; resubmitting', request_id)
        remote = self.client.submit('reanalysis-era5-land', part.request)
        self.requests[part.key] = remote.request_id
        # Persist before waiting/downloading, independently of raw-cache publication.
        self._save()
        return remote

    def ready(self, wanted: Collection[str]) -> str:
        while True:
            for key, remote in self.pending.items():
                if key in wanted and remote.results_ready:
                    return key
            time.sleep(10)

    def download(self, key: str, path: Path) -> None:
        remote = self.pending[key]
        pending = path.with_name(path.name + '.part')
        LOGGER.info('Downloading ERA5-Land GRIB: %s', path.name)
        started = time.monotonic()
        # Use the CDS session for file downloads as well as API requests.
        remote.download_options['session'] = self.client.session
        remote.download(str(pending))
        pending.replace(path)
        del self.pending[key]
        LOGGER.info('Downloaded %s in %.1f s', path.name, time.monotonic() - started)
