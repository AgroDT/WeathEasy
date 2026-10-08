from typing import Annotated, NotRequired, TypedDict, cast

from anyio import CapacityLimiter
from fastapi import Depends, Request

from .config import Settings
from .models import Variables
from .repository import WeatherRepository
from weatheasy.const import CFS2_BANDS, CMIP6_VARS, CORRECTION_INFO, ERA5_ALL_VARS
from weatheasy.query import get_cfs2_data, get_cmip6_data, get_era5_data
from weatheasy.storage import get_store


class AppState(TypedDict):
    settings: Settings
    variables: Variables
    cfs2_repository: NotRequired[WeatherRepository]
    cmip6_repository: NotRequired[WeatherRepository]
    era5_repository: NotRequired[WeatherRepository]


def create_state(settings: Settings, limiter: CapacityLimiter) -> AppState:
    sources = settings.sources
    state = cast('AppState', {'settings': settings})
    variables = {}

    if sources.cfs2:
        variables['cfs2'] = {k: v.info for k, v in CFS2_BANDS.items()} | CORRECTION_INFO
        state['cfs2_repository'] = WeatherRepository(
            get_store(sources.cfs2.store), get_cfs2_data, limiter
        )
    if sources.cmip6:
        variables['cmip6'] = CMIP6_VARS
        state['cmip6_repository'] = WeatherRepository(
            get_store(sources.cmip6.store), get_cmip6_data, limiter
        )
    if sources.era5:
        variables['era5'] = ERA5_ALL_VARS | CORRECTION_INFO
        state['era5_repository'] = WeatherRepository(
            get_store(sources.era5.store), get_era5_data, limiter
        )

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
