"""Local working files with optional persistent S3 backing."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem

from weatheasy.file_storage import FileStore, get_file_store


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class DownloadCache:
    """Reuse working paths and publish files as soon as downloads complete."""

    def __init__(
        self, root: Path, remote: FileStore | None = None, *, temporary: bool = False
    ) -> None:
        self.root = root
        self.remote = remote
        self.temporary = temporary
        self._new: set[Path] = set()

    def exists(self, name: str) -> bool:
        """Check availability without fetching a remote working copy."""
        return (self.root / name).is_file() or (
            self.remote is not None and self.remote.exists(name)
        )

    def file(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() and self.remote is not None:
            if self.remote.exists(name):
                pending = path.with_suffix(path.suffix + '.fetching')
                try:
                    with self.remote.open(name) as source, pending.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    pending.replace(path)
                finally:
                    pending.unlink(missing_ok=True)
            else:
                self._new.add(path)
        return path

    def publish(self, paths: Iterable[Path]) -> None:
        """Upload new downloaded files, keeping their local working copies."""
        if self.remote is None:
            return
        for path in paths:
            if path in self._new:
                name = path.relative_to(self.root).as_posix()
                with path.open('rb') as source, self.remote.write_atomic(name) as target:
                    shutil.copyfileobj(source, target)
                self._new.remove(path)


@contextmanager
def download_cache(location: str | Path | FileStore | None, *parts: str) -> Iterator[DownloadCache]:
    """Use local directories directly; stage S3 files on demand in a temporary directory."""
    store = (
        (location if isinstance(location, FileStore) else get_file_store(location))
        if location is not None
        else None
    )
    if store is not None:
        store = store.child(*parts)
        if isinstance(store.fs, LocalFileSystem):
            root = Path(store.root)
            root.mkdir(parents=True, exist_ok=True)
            yield DownloadCache(root)
            return
    with TemporaryDirectory() as temporary:
        yield DownloadCache(Path(temporary), store, temporary=True)
