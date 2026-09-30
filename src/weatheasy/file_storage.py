"""Prepared local/S3 file access and complete-file publication."""

from __future__ import annotations

import posixpath
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fsspec.implementations.local import LocalFileSystem

from weatheasy.error import S3ImportError


if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike
    from typing import BinaryIO

    from fsspec import AbstractFileSystem


@dataclass(frozen=True)
class FileStore:
    """A filesystem and root already resolved by the caller."""

    fs: AbstractFileSystem
    root: str

    def child(self, *parts: str) -> FileStore:
        return FileStore(self.fs, posixpath.join(self.root, *parts))

    def open(self, name: str = '', mode: str = 'rb') -> BinaryIO:
        path = posixpath.join(self.root, name) if name else self.root
        return cast('BinaryIO', self.fs.open(path, mode))

    def exists(self, name: str = '') -> bool:
        path = posixpath.join(self.root, name) if name else self.root
        return self.fs.exists(path)

    def glob(self, pattern: str) -> list[str]:
        return self.fs.glob(posixpath.join(self.root, pattern))

    @contextmanager
    def write_atomic(self, name: str) -> Iterator[BinaryIO]:
        """Expose one complete file after writing succeeds; callers serialize writers."""
        if isinstance(self.fs, LocalFileSystem):
            destination = Path(self.root) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A same-directory replacement stays atomic across filesystem mounts.
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix='.tmp', delete=False
            ) as temporary_stream:
                temporary = Path(temporary_stream.name)
            try:
                with temporary.open('wb') as stream:
                    yield stream
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            with self.fs.transaction, self.open(name, 'wb') as stream:
                yield stream


def get_file_store(
    uri: str | PathLike[str], *, filesystem: AbstractFileSystem | None = None
) -> FileStore:
    """Resolve a local file/directory or S3 URI at an application boundary.

    Supply a synchronous S3 filesystem to share its client with other files
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
        return FileStore(filesystem, location[5:])
    if filesystem is not None:
        msg = 'An S3 filesystem can only be used with an s3:// URI'
        raise ValueError(msg)
    return FileStore(LocalFileSystem(auto_mkdir=True), str(Path(location).expanduser()))
