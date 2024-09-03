from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

from weatheasy.error import S3ImportError
from weatheasy.version import __version__


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

    if root.startswith('s3://'):
        try:
            import s3fs
        except ImportError as exc:
            raise S3ImportError from exc
        else:
            store = s3fs.S3Map(root[5:], s3fs.S3FileSystem())
    else:
        store = root

    return zarr.group(store)


def init_parser(module: str = __package__) -> ArgumentParser:
    version = __version__ or 'unknown version'
    parser = ArgumentParser(module, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=version)
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
