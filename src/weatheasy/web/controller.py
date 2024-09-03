from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from anyio import to_thread
from fastapi.responses import StreamingResponse

from .config import get_config
from .models import DataQuery, Variables, VarInfo
from weatheasy.const import CFS2_BANDS, CMIP6_VARS, ONE_DAY


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    import numpy as np
    import zarr
    from numpy.typing import NDArray

    from weatheasy.util import FormatFloat

    type Getter = Callable[..., NDArray[np.float32]]


def get_variables() -> Variables:
    return Variables(
        cfs2={k: VarInfo(en=v.info.en, ru=v.info.ru) for k, v in CFS2_BANDS.items()},
        cmip6={k: VarInfo(en=v.en, ru=v.ru) for k, v in CMIP6_VARS.items()},
    )


async def get_data(getter: Getter, query: DataQuery) -> StreamingResponse:
    cfg = get_config()
    data = await to_thread.run_sync(_exec_getter, getter, query, cfg.storage)
    content = _stream_data(data, query, cfg.format_float)
    return StreamingResponse(content, media_type='application/json')


def _exec_getter(getter: Getter, query: DataQuery, root: zarr.Group):
    data = getter(**query, root=root)
    return data.transpose()


def _stream_data(data: NDArray[np.float32], query: DataQuery, format_float: FormatFloat):
    sio = StringIO()
    format_row = _row_formatter_factory(sio, query['variables'], format_float)
    date_ = query['begin']
    buf_size = 10240
    sio.write('[')
    format_row(date_, data[0])
    for row in data[1:]:
        date_ += ONE_DAY
        sio.write(',')
        format_row(date_, row)
        pos = sio.tell()
        if pos >= buf_size:
            yield sio.getvalue()[:pos]
            sio.seek(0)
    sio.write(']')
    yield sio.getvalue()[: sio.tell()]


def _row_formatter_factory(sio: StringIO, variables: Sequence[str], format_float: FormatFloat):
    def format_row(date_: date, row: NDArray[np.float32]):
        sio.write('{')
        for var, val in zip(variables, row, strict=False):
            sio.write(f'"{var}":')
            sio.write(format_float(val))
            sio.write(',')
        sio.write(f'"date":"{date_}"}}')

    return format_row
