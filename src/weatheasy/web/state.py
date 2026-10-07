from typing import Annotated, TypedDict

from fastapi import Depends, Request

from .config import Settings
from .models import Variables


class AppState(TypedDict):
    settings: Settings
    variables: Variables


def get_settings(request: Request) -> Settings:
    return request.state['settings']


def get_variables(request: Request) -> Variables:
    return request.state['variables']


type SettingsDependency = Annotated[Settings, Depends(get_settings)]
type VariablesDependency = Annotated[Variables, Depends(get_variables)]
