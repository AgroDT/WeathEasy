"""Resolve command locations while sharing one S3 filesystem."""

from __future__ import annotations

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

    def group(self, uri: str) -> Group:
        return get_storage(uri, filesystem=self.async_fs if uri.startswith('s3://') else None)
