"""Resolve command locations while sharing one S3 filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

from weatheasy.file_storage import get_file_store
from weatheasy.storage import get_store
from weatheasy.util import get_storage


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from zarr import Group
    from zarr.abc.store import Store

    from weatheasy.file_storage import FileStore


class Storage:
    def __init__(self, *locations: str | None) -> None:
        uri = next((value for value in locations if value and value.startswith('s3://')), None)
        self.filesystem: AbstractFileSystem | None = get_file_store(uri).fs if uri else None
        self.async_fs = (
            AsyncFileSystemWrapper(fs=self.filesystem, asynchronous=True)
            if self.filesystem is not None
            else None
        )

    def files(self, uri: str) -> FileStore:
        return get_file_store(uri, filesystem=self.filesystem if uri.startswith('s3://') else None)

    def store(self, uri: str) -> Store:
        return get_store(uri, filesystem=self.async_fs if uri.startswith('s3://') else None)

    def dem_source(self, uri: str, *, headers_file: str | None = None) -> Store:
        """Read an HTTP DEM using headers/netrc, or existing local/S3 storage."""
        if not uri.startswith(('https://', 'http://')):
            if headers_file is not None:
                msg = '--dem-headers-file requires an HTTP or HTTPS --dem-source'
                raise ValueError(msg)
            return self.store(uri)
        from fsspec.implementations.http import HTTPFileSystem  # noqa: PLC0415
        from zarr.storage import FsspecStore  # noqa: PLC0415

        headers = {}
        if headers_file is not None:
            try:
                headers = json.loads(Path(headers_file).read_text())
            except (ValueError, UnicodeError):
                msg = 'DEM headers file must contain a JSON object of string headers'
                raise ValueError(msg) from None
            if not isinstance(headers, dict) or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or '\r' in key
                or '\n' in key
                or '\r' in value
                or '\n' in value
                for key, value in headers.items()
            ):
                msg = 'DEM headers file must contain a JSON object of string headers'
                raise ValueError(msg)
        filesystem = HTTPFileSystem(
            asynchronous=True,
            client_kwargs={'trust_env': not any(key.lower() == 'authorization' for key in headers)},
            headers=headers,
        )
        return FsspecStore(filesystem, path=uri.rstrip('/'), read_only=True)

    def group(self, uri: str) -> Group:
        return get_storage(uri, filesystem=self.async_fs if uri.startswith('s3://') else None)
