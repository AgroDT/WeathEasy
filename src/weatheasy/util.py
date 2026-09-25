from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr


if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem


try:
    from weatheasy._version import version as _version
except ImportError:
    _version = 'unknown version'


type FormatFloat = Callable[[np.floating], str]


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def get_storage(root: str | Path, *, filesystem: AsyncFileSystem | None = None) -> zarr.Group:
    """Initialize zarr group.

    To work with S3 pass `root` in format `s3://<bucket>[/path]`.

    S3 credentials should be passed with standard AWS environment variables:

    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_ENDPOINT_URL_S3
    - etc.

    Supply ``filesystem`` to share one asynchronous S3 filesystem between
    this weather store and other Zarr stores in the same job.
    """
    from weatheasy.storage import get_store  # noqa: PLC0415

    return zarr.open_group(get_store(root, filesystem=filesystem), mode='a', zarr_format=3)


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
