from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
from anyio import to_thread
from fastapi.responses import StreamingResponse

from .models import DataQuery, Variables
from weatheasy.const import CFS2_BANDS, CMIP6_VARS, CORRECTION_INFO, ERA5_ALL_VARS, ONE_DAY


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .config import Settings, SourcesSettings
    from weatheasy.util import FormatFloat

    type Getter = Callable[..., NDArray[np.float32]]


def build_variables(sources: SourcesSettings) -> Variables:
    data = {
        'cfs2': sources.cfs2 and {k: v.info for k, v in CFS2_BANDS.items()} | CORRECTION_INFO,
        'cmip6': sources.cmip6 and CMIP6_VARS,
        'era5': sources.era5 and ERA5_ALL_VARS | CORRECTION_INFO,
    }

    return Variables.model_validate(data, from_attributes=True)


async def get_data(
    store: Store,
    getter: Getter,
    query: DataQuery,
    settings: Settings,
) -> StreamingResponse:
    data = await to_thread.run_sync(_exec_getter, getter, query, store)
    content = _stream_data(data, query, settings.format_float)
    return StreamingResponse(content, media_type='application/json')


def _exec_getter(getter: Getter, query: DataQuery, store: Store):
    data = getter(**query, store=store)
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
            sio.write(format_float(np.float32(val)))
            sio.write(',')
        sio.write(f'"date":"{date_}"}}')

    return format_row
