from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
from fastapi.responses import StreamingResponse

from weatheasy.const import ONE_DAY


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from numpy.typing import NDArray

    from .config import Settings
    from .models import DataQuery
    from .repository import WeatherRepository
    from weatheasy.util import FormatFloat


async def data_response(
    repository: WeatherRepository,
    query: DataQuery,
    settings: Settings,
) -> StreamingResponse:
    data = await repository.read(query)
    content = _stream_data(data.transpose(), query, settings.format_float)
    return StreamingResponse(content, media_type='application/json')


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
            sio.write(format_float(np.float32(val)))
            sio.write(',')
        sio.write(f'"date":"{date_}"}}')

    return format_row
