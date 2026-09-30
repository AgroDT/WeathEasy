"""Prepared filesystem access to a model's multi-file artifact directory."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fsspec.implementations.local import LocalFileSystem

from weatheasy.error import S3ImportError


if TYPE_CHECKING:
    from os import PathLike
    from typing import BinaryIO

    from fsspec import AbstractFileSystem


@dataclass(frozen=True)
class Artifacts:
    """A filesystem and root already resolved by the caller."""

    fs: AbstractFileSystem
    root: str

    def child(self, *parts: str) -> Artifacts:
        return Artifacts(self.fs, posixpath.join(self.root, *parts))

    def open(self, name: str, mode: str = 'rb') -> BinaryIO:
        return cast('BinaryIO', self.fs.open(posixpath.join(self.root, name), mode))

    def exists(self, name: str) -> bool:
        return self.fs.exists(posixpath.join(self.root, name))


def get_artifacts(
    uri: str | PathLike[str], *, filesystem: AbstractFileSystem | None = None
) -> Artifacts:
    """Resolve a local directory or S3 URI at an application boundary.

    Supply a synchronous S3 filesystem to share its client with other artifacts
    and Zarr stores accessed through an AsyncFileSystemWrapper.
    """
    location = str(uri)
    if location.startswith('s3://'):
        if not location[5:].split('/', maxsplit=1)[0]:
            msg = 'S3 URI requires a bucket'
            raise ValueError(msg)
        if filesystem is None:
            try:
                import s3fs  # noqa: PLC0415
            except ImportError as exc:
                raise S3ImportError from exc
            filesystem = s3fs.S3FileSystem()
        return Artifacts(filesystem, location[5:])
    if filesystem is not None:
        msg = 'An S3 filesystem can only be used with an s3:// URI'
        raise ValueError(msg)
    return Artifacts(LocalFileSystem(auto_mkdir=True), str(Path(location).expanduser()))
