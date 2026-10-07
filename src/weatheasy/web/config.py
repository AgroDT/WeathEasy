from functools import cached_property
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from zarr.abc.store import Store

from weatheasy.storage import get_store
from weatheasy.util import FormatFloat, float_formatter_factory


class SourceSettings(BaseModel):
    store: Annotated[str, Field(min_length=1)]

    @cached_property
    def zarr_store(self) -> Store:
        return get_store(self.store)


class SourcesSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = {'extra': 'forbid'}

    cfs2: SourceSettings | None = None
    era5: SourceSettings | None = None
    cmip6: SourceSettings | None = None


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = {
        'env_prefix': 'WEATHEASY__',
        'env_nested_delimiter': '__',
        'env_file': ('.env', '.local.env'),
        'extra': 'ignore',
    }

    sources: Annotated[SourcesSettings, Field(default_factory=SourcesSettings)]
    precision: PositiveInt = 6
    enable_cors: bool = False

    @computed_field
    @cached_property
    def format_float(self) -> FormatFloat:
        return float_formatter_factory('null', self.precision)
