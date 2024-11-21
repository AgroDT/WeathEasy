from functools import cache, cached_property

import zarr
from pydantic import PositiveInt, computed_field
from pydantic_settings import BaseSettings

from weatheasy.util import FormatFloat, float_formatter_factory, get_storage


class Settings(BaseSettings):
    model_config = {
        'env_prefix': 'WEATHEASY__',
        'env_nested_delimiter': '__',
        'env_file': ('.env', '.local.env'),
        'extra': 'ignore',
    }

    data_root: str
    precision: PositiveInt = 6
    enable_cors: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def storage(self) -> zarr.Group:
        return get_storage(self.data_root)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def format_float(self) -> FormatFloat:
        return float_formatter_factory('null', self.precision)


@cache
def get_config() -> Settings:
    return Settings()
