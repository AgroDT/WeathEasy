from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import date

    from rasterio.coords import BoundingBox

    from weatheasy import Coords


class S3ImportError(ImportError):
    def __init__(self) -> None:
        super().__init__('To work with S3 storage install s3fs or weatheasy[s3]', name='s3fs')


class CFS2Error(RuntimeError):
    def __init__(self) -> None:
        super().__init__('CFS2 datasets not found')


class BaseValueError(ValueError): ...


class CoordsError(BaseValueError):
    def __init__(self, coords: Coords, bbox: BoundingBox) -> None:
        super().__init__(f'{coords} are out of {bbox}')


class DateRangeError(BaseValueError):
    def __init__(self) -> None:
        super().__init__('first date must be less than or equal to last')


class CMIP6DateRangeError(BaseValueError):
    def __init__(self, date_: date, first: date, last: date) -> None:
        super().__init__(f'date {date_} is out of range [{first}; {last}]')
