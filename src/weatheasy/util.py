from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import FsspecStore

from weatheasy.error import S3ImportError


try:
    from weatheasy._version import version as _version
except ImportError:
    _version = 'unknown version'


type FormatFloat = Callable[[np.floating], str]


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def get_storage(root: str) -> zarr.Group:
    """Initialize zarr group.

    To work with S3 pass `root` in format `s3://<bucket>[/path]`.

    S3 credentials should be passed with standard AWS environment variables:

    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_ENDPOINT_URL_S3
    - etc.
    """

    store: str | FsspecStore
    if root.startswith('s3://'):
        try:
            import s3fs
        except ImportError as exc:
            raise S3ImportError from exc
        else:
            store = FsspecStore(s3fs.S3FileSystem(asynchronous=True), path=root[5:])
    else:
        store = root

    return zarr.open_group(store, mode='a', zarr_format=3)


def init_parser(module: str = 'weatheasy') -> ArgumentParser:
    parser = ArgumentParser(module, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=_version)
    parser.add_argument(
        '-d',
        '--data',
        default=Path.cwd().joinpath('weatheasy', 'zarr').as_posix(),
        metavar='STORE',
        help='Zarr store for downloaded data',
    )

    return parser


def float_formatter_factory(nan: str, precision: int) -> FormatFloat:
    precision = int(precision)
    if not 0 < precision <= 6:
        msg = f'precision must be a positive integer, got "{precision}"'
        raise ValueError(msg)

    def impl(value: np.floating) -> str:
        if np.isnan(value):
            return nan
        return np.format_float_positional(value, precision, trim='-')

    return impl
