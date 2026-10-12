"""Local working files with optional persistent S3 backing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem

from weatheasy.file_storage import FileStore, get_file_store


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable


def default_work_dir() -> Path:
    """Keep retryable downloads on disk, outside the system temporary directory."""

    def get_state_from_env(name: str, *parts: str) -> Path:
        if raw := os.environ.get(name):
            return Path(raw)
        return Path.home().joinpath(*parts)

    if sys.platform == 'win32':
        state = get_state_from_env('LOCALAPPDATA', 'AppData', 'Local')
    elif sys.platform == 'darwin':
        state = Path.home().joinpath('Library', 'Application Support')
    else:
        state = get_state_from_env('XDG_STATE_HOME', '.local', 'state')

    return state.joinpath('weatheasy', 'downloads')


class DownloadCache:
    """Reuse working paths and publish files as soon as downloads complete."""

    def __init__(
        self, root: Path, remote: FileStore | None = None, *, temporary: bool = False
    ) -> None:
        self.root = root
        self.remote = remote
        self.temporary = temporary

    def child(self, *parts: str) -> DownloadCache:
        """Use a subdirectory while retaining the working-file lifecycle."""
        root = self.root.joinpath(*parts)
        root.mkdir(parents=True, exist_ok=True)
        return DownloadCache(
            root, self.remote.child(*parts) if self.remote else None, temporary=self.temporary
        )

    def exists(self, name: str) -> bool:
        """Check availability without fetching a remote working copy."""
        return (self.root / name).is_file() or (
            self.remote is not None and self.remote.exists(name)
        )

    def file(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() and self.remote is not None and self.remote.exists(name):
            pending = path.with_name(path.name + '.part')
            try:
                with self.remote.open(name) as source, pending.open('wb') as target:
                    shutil.copyfileobj(source, target)
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
        if path.is_file() and self.remote is not None:
            path.touch()
        return path

    def write(self, path: Path, content: bytes) -> None:
        """Expose a provider download only after the complete file is written."""
        pending = path.with_name(path.name + '.part')
        try:
            pending.write_bytes(content)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)

    def publish(self, paths: Iterable[Path]) -> None:
        """Publish complete downloads, also retrying uploads from earlier processes."""
        if self.remote is None:
            return
        for path in paths:
            name = path.relative_to(self.root).as_posix()
            if not self.remote.exists(name):
                with path.open('rb') as source, self.remote.write_atomic(name) as target:
                    shutil.copyfileobj(source, target)

    def release(self, paths: Iterable[Path]) -> None:
        """Discard working copies after their results commit; retain local caches."""
        if self.remote is not None or self.temporary:
            for path in paths:
                path.unlink(missing_ok=True)

    def cleanup(self, needed: Callable[[str], bool], *, max_age_days: int = 7) -> None:
        """Check job requirements before removing old files or abandoned partials."""
        if max_age_days < 0:
            msg = 'work_max_age_days must be non-negative'
            raise ValueError(msg)
        if self.remote is None:
            return
        cutoff = time.time() - max_age_days * 86400
        directories = []
        for path in self.root.rglob('*'):
            if path.is_dir():
                directories.append(path)
                continue
            if not path.is_file():
                continue
            name = path.relative_to(self.root).as_posix()
            if name == '.weatheasy-cache.json':
                continue
            # Partial files never constitute reusable downloads, even when needed.
            if name.endswith('.part') or (not needed(name) and path.stat().st_mtime < cutoff):
                path.unlink()
        for directory in sorted(directories, reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()


@contextmanager
def download_cache(
    location: str | Path | FileStore | None,
    *parts: str,
    work_dir: str | Path | None = None,
    target: str = '',
) -> Generator[DownloadCache]:
    """Use local caches directly and retain remote working copies across retries."""
    store = (
        (location if isinstance(location, FileStore) else get_file_store(location))
        if location is not None
        else None
    )
    if store is not None:
        if isinstance(store.fs, LocalFileSystem):
            root = Path(store.root).joinpath(*parts)
            root.mkdir(parents=True, exist_ok=True)
            yield DownloadCache(root)
            return
        # Separate backends and result stores without including credentials.
        settings = {
            'protocol': store.fs.protocol,
            'download_cache': store.root,
            'endpoint': getattr(store.fs, 'client_kwargs', {}).get('endpoint_url')
            or os.environ.get('AWS_ENDPOINT_URL_S3')
            or os.environ.get('AWS_ENDPOINT_URL'),
            'target': target,
        }
        identity = json.dumps(settings, sort_keys=True)
        namespace = hashlib.sha256(identity.encode()).hexdigest()[:20]
        base = Path(work_dir).expanduser() if work_dir is not None else default_work_dir()
        root = base.joinpath(*(parts[:1]), namespace, *(parts[1:]))
        root.mkdir(parents=True, exist_ok=True)
        cache = DownloadCache(root, store.child(*parts))
        cache.write(
            root / '.weatheasy-cache.json',
            (json.dumps({**settings, 'parts': parts}, indent=2) + '\n').encode(),
        )
        yield cache
        return
    with TemporaryDirectory() as temporary:
        yield DownloadCache(Path(temporary), temporary=True)
