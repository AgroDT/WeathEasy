from typing import Annotated, NotRequired, TypedDict, cast

from anyio import CapacityLimiter
from fastapi import Depends, Request

from .config import Settings
from .models import Variables
from .repository import Cfs2Repository, Cmip6Repository, Era5Repository, WeatherRepository
from weatheasy.const import CFS2_BANDS, CMIP6_VARS, CORRECTION_INFO, ERA5_ALL_VARS
from weatheasy.storage import get_store


class AppState(TypedDict):
    settings: Settings
    variables: Variables
    cfs2_repository: NotRequired[WeatherRepository]
    cmip6_repository: NotRequired[WeatherRepository]
    era5_repository: NotRequired[WeatherRepository]


def create_state(settings: Settings, limiter: CapacityLimiter) -> AppState:
    sources = settings.sources
    read_timeout = settings.storage_read_timeout
    operation_timeout = settings.storage_operation_timeout
    state = cast('AppState', {'settings': settings})
    variables = {}

    def create_repo(store: str, repository: type[WeatherRepository]):
        return repository(
            get_store(store),
            limiter,
            read_timeout=read_timeout,
            operation_timeout=operation_timeout,
        )

    if sources.cfs2:
        variables['cfs2'] = {k: v.info for k, v in CFS2_BANDS.items()} | CORRECTION_INFO
        state['cfs2_repository'] = create_repo(sources.cfs2.store, Cfs2Repository)
    if sources.cmip6:
        variables['cmip6'] = CMIP6_VARS
        state['cmip6_repository'] = create_repo(sources.cmip6.store, Cmip6Repository)
    if sources.era5:
        variables['era5'] = ERA5_ALL_VARS | CORRECTION_INFO
        state['era5_repository'] = create_repo(sources.era5.store, Era5Repository)

    state['variables'] = Variables.model_validate(variables, from_attributes=True)

    return state


async def _get_settings(request: Request) -> Settings:
    return request.state['settings']


async def _get_variables(request: Request) -> Variables:
    return request.state['variables']


async def _get_cfs2_repository(request: Request) -> WeatherRepository:
    return request.state['cfs2_repository']


async def _get_cmip6_repository(request: Request) -> WeatherRepository:
    return request.state['cmip6_repository']


async def _get_era5_repository(request: Request) -> WeatherRepository:
    return request.state['era5_repository']


type SettingsDependency = Annotated[Settings, Depends(_get_settings)]
type VariablesDependency = Annotated[Variables, Depends(_get_variables)]
type CFS2RepositoryDependency = Annotated[WeatherRepository, Depends(_get_cfs2_repository)]
type CMIP6RepositoryDependency = Annotated[WeatherRepository, Depends(_get_cmip6_repository)]
type ERA5RepositoryDependency = Annotated[WeatherRepository, Depends(_get_era5_repository)]
