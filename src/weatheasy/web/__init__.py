from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .controller import build_variables
from .routes import cfs2_router, cmip6_router, common_router, era5_router
from .state import AppState
from weatheasy import __version__
from weatheasy.error import BaseValueError


async def handle_value_error(_request: Request, err: BaseValueError) -> JSONResponse:
    return JSONResponse({'detail': str(err)}, 422)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an API for the configured sources without opening their stores."""
    settings = settings if settings is not None else Settings()
    sources = settings.sources

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[AppState]:
        yield {'settings': settings, 'variables': build_variables(settings.sources)}

    app = FastAPI(
        title='WeathEasy',
        version=__version__ or 'unknown',
        exception_handlers={BaseValueError: handle_value_error},
        lifespan=lifespan,
    )
    if settings.enable_cors:
        app.add_middleware(CORSMiddleware, allow_methods='*', allow_origin_regex='.*')

    app.include_router(common_router)
    if sources.cfs2:
        app.include_router(cfs2_router)
    if sources.cmip6:
        app.include_router(cmip6_router)
    if sources.era5:
        app.include_router(era5_router)

    return app
