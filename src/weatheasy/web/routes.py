from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from . import controller as ctr, models as mls
from .state import (
    CFS2RepositoryDependency,
    CMIP6RepositoryDependency,
    ERA5RepositoryDependency,
    SettingsDependency,
    VariablesDependency,
)


common_router = APIRouter()
cfs2_router = APIRouter(tags=['CFS2'])
era5_router = APIRouter(tags=['ERA5'])
cmip6_router = APIRouter(tags=['CMIP6'])


@common_router.get('/startupz', tags=['Health'])
@common_router.get('/readyz', tags=['Health'])
@common_router.get('/livez', tags=['Health'])
async def get_health() -> dict[str, str]:
    """Check API responsiveness without accessing weather storage."""
    return {'status': 'ok'}


@common_router.get('/variables', response_model=mls.Variables, response_model_exclude_none=True)
async def get_variables(variables: VariablesDependency) -> mls.Variables:
    return variables


@cfs2_router.get(
    '/cfs2',
    response_model=list[mls.CFS2DataItem],
    response_model_exclude_unset=True,
)
async def get_cfs2_data(
    query: mls.SFS2Query, settings: SettingsDependency, repository: CFS2RepositoryDependency
) -> StreamingResponse:
    return await ctr.data_response(repository, query, settings)


@cmip6_router.get(
    '/cmip6',
    response_model=list[mls.CMIP6DataItem],
    response_model_exclude_unset=True,
)
async def get_cmip6_data(
    query: mls.CMIP6Query, settings: SettingsDependency, repository: CMIP6RepositoryDependency
) -> StreamingResponse:
    return await ctr.data_response(repository, query, settings)


@era5_router.get(
    '/era5',
    response_model=list[mls.ERA5DataItem],
    response_model_exclude_unset=True,
)
async def get_era5_data(
    query: mls.ERA5Query, settings: SettingsDependency, repository: ERA5RepositoryDependency
) -> StreamingResponse:
    return await ctr.data_response(repository, query, settings)
