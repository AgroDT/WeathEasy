from functools import cache

from pydantic import PositiveInt
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        'env_prefix': 'WEATHEASY__',
        'env_nested_delimiter': '__',
        'env_file': ('.env', '.local.env'),
        'extra': 'ignore',
    }

    data_root: str
    decimal_places: PositiveInt = 6


@cache
def get_config() -> Settings:
    return Settings()
