from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
from anyio import to_thread
from fastapi.responses import StreamingResponse

from .config import get_config
from .models import DataQuery, Variables, VarInfo
from weatheasy.const import CFS2_BANDS, CMIP6_VARS, ONE_DAY
from weatheasy.util import get_storage


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from numpy.typing import NDArray

    Getter = Callable[..., NDArray[np.float32]]


def get_variables() -> Variables:
    return Variables(
        cfs2={k: VarInfo(en=v.info.en, ru=v.info.ru) for k, v in CFS2_BANDS.items()},
        cmip6={k: VarInfo(en=v.en, ru=v.ru) for k, v in CMIP6_VARS.items()},
    )


async def get_data(getter: Getter, query: DataQuery) -> StreamingResponse:
    cfg = get_config()
    data = await to_thread.run_sync(_exec_getter, getter, query, cfg.data_root)
    content = _stream_data(data, query, cfg.decimal_places)
    return StreamingResponse(content, media_type='application/json')


def _exec_getter(getter: Getter, query: DataQuery, data_root: str):
    data = getter(**query, root=get_storage(data_root))
    return data.transpose()


def _stream_data(data: NDArray[np.float32], query: DataQuery, precision: int):
    sio = StringIO()
    format_row = _row_formatter_factory(sio, query['variables'], precision)
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


def _row_formatter_factory(sio: StringIO, variables: Sequence[str], precision: int):
    def format_row(date_: date, row: NDArray[np.float32]):
        sio.write('{')
        for var, val in zip(variables, row, strict=False):
            sio.write(f'"{var}":')
            sio.write(
                'null' if np.isnan(val) else np.format_float_positional(val, precision, trim='-')
            )
            sio.write(',')
        sio.write(f'"date":"{date_}"}}')

    return format_row
