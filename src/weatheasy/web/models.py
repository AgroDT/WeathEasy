from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Annotated, TypeAlias, TypedDict

from fastapi import Depends, Query
from pydantic import BaseModel, Field, create_model

from weatheasy import Coords, const


if TYPE_CHECKING:
    CFS2Var: TypeAlias = Enum
    CMIP6Var: TypeAlias = Enum
else:
    CFS2Var = Enum('CFS2Var', {k: k for k in const.CFS2_BANDS})
    CMIP6Var = Enum('CMIP6Var', {k: k for k in const.CMIP6_VARS})


class VarInfo(BaseModel):
    en: str
    ru: str


class Variables(BaseModel):
    cfs2: dict[str, VarInfo]
    cmip6: dict[str, VarInfo]


class DataQuery(TypedDict):
    coords: Coords
    begin: date
    end: date
    variables: list[str]


def _query_base(
    lat: Annotated[float, Query(description='EPSG:4326', ge=-180, le=180, example=55.75222)],
    lon: Annotated[float, Query(description='EPSG:4326', ge=-90, le=90, example=37.61556)],
    begin: Annotated[date, Query()],
    end: Annotated[date, Query()],
):
    return {
        'coords': Coords(latitude=lat, longitude=lon),
        'begin': begin,
        'end': end,
    }


_QueryBase = Annotated[dict, Depends(_query_base)]


def _cfs2_query(
    query: _QueryBase,
    variables: Annotated[set[CFS2Var], Query(alias='var')],
):
    query['variables'] = [v.value for v in variables]
    return query


def _cmip6_query(
    query: _QueryBase,
    variables: Annotated[set[CMIP6Var], Query(alias='var')],
):
    query['variables'] = [v.value for v in variables]
    return query


def _data_item(name: str, variables: type[Enum]):
    field_definitions: dict = {k.value: (DecimalField | None, None) for k in variables}
    field_definitions['date_'] = DateField, ...
    return create_model(name, **field_definitions)


SFS2Query = Annotated[DataQuery, Depends(_cfs2_query)]
CMIP6Query = Annotated[DataQuery, Depends(_cmip6_query)]
DateField = Annotated[date, Field(alias='date')]
DecimalField = Annotated[float | None, Field()]
CFS2DataItem = _data_item('CFS2DataItem', CFS2Var)
CMIP6DataItem = _data_item('CMIP6DataItem', CMIP6Var)
