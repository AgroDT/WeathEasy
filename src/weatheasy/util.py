from __future__ import annotations

from datetime import UTC, datetime

import zarr

from weatheasy.error import S3ImportError


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
