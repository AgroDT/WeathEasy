from __future__ import annotations

from contextlib import aclosing
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
from fastapi.responses import StreamingResponse

from weatheasy.const import ONE_DAY


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence
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
    batches = repository.stream(query)
    first = await anext(batches)
    content = _stream_data(first, batches, query, settings.format_float)
    return StreamingResponse(content, media_type='application/json')


async def _stream_data(
    first: NDArray[np.float32],
    batches: AsyncGenerator[NDArray[np.float32]],
    query: DataQuery,
    format_float: FormatFloat,
) -> AsyncIterator[str]:
    sio = StringIO()
    format_row = _row_formatter_factory(sio, query['variables'], format_float)
    date_ = query['begin']
    first_row = True
    data = first
    sio.write('[')
    async with aclosing(batches):
        while True:
            for row in data.transpose():
                if not first_row:
                    sio.write(',')
                format_row(date_, row)
                first_row = False
                if date_ < query['end']:
                    date_ += ONE_DAY
                if sio.tell() >= 10240:
                    yield sio.getvalue()
                    sio.seek(0)
                    sio.truncate()
            if sio.tell():
                # Flush each storage batch before starting the next read.
                yield sio.getvalue()
                sio.seek(0)
                sio.truncate()
            data = await anext(batches, None)
            if data is None:
                break
    yield ']'


def _row_formatter_factory(sio: StringIO, variables: Sequence[str], format_float: FormatFloat):
    def format_row(date_: date, row: NDArray[np.float32]):
        sio.write('{')
        for var, val in zip(variables, row, strict=False):
            sio.write(f'"{var}":')
            sio.write(format_float(np.float32(val)))
            sio.write(',')
        sio.write(f'"date":"{date_}"}}')

    return format_row
