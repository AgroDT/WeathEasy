from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import weatheasy
from . import controller as ctr, models as mls
from .config import get_config
from weatheasy.error import BaseValueError
from weatheasy.version import __version__


async def handle_value_error(_request: Request, err: BaseValueError) -> JSONResponse:
    return JSONResponse({'detail': str(err)}, 422)


app = FastAPI(
    title='WeathEasy',
    version=__version__ or 'unknown',
    exception_handlers={
        BaseValueError: handle_value_error,
    },
)


if get_config().enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_methods='*',
        allow_origin_regex='.*',
    )


@app.get('/variables', response_model=mls.Variables)
def get_variables() -> mls.Variables:
    return ctr.get_variables()


@app.get(
    path='/cfs2',
    response_model=list[mls.CFS2DataItem],
    response_model_exclude_unset=True,
)
async def get_cfs2_data(query: mls.SFS2Query) -> StreamingResponse:
    return await ctr.get_data(weatheasy.get_cfs2_data, query)


@app.get(
    path='/cmip6',
    response_model=list[mls.CMIP6DataItem],
    response_model_exclude_unset=True,
)
async def get_cmip6_data(query: mls.CMIP6Query) -> StreamingResponse:
    return await ctr.get_data(weatheasy.get_cmip6_data, query)
