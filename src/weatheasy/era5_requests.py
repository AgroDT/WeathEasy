"""Build ERA5-Land requests and stable raw-cache identities without doing I/O."""

from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass
from datetime import date

from weatheasy import const


EXTRA = {
    'd2m': '2m_dewpoint_temperature',
    'ssr': 'surface_net_solar_radiation',
    'str': 'surface_net_thermal_radiation',
}


def next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


@dataclass(frozen=True)
class Part:
    month: date
    boundary: bool
    request: dict
    required: set[str]
    key: str

    @property
    def filename(self) -> str:
        return f'{self.month:%Y%m}-{self.key}.grib'


def month_parts(month: date, bbox: tuple, *, boundary: bool = False) -> list[Part]:
    if boundary:
        return [_accumulated_part(month, bbox, boundary=True)]
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values() if not v.accumulated}
    names[EXTRA['d2m']] = 'd2m'
    variables = sorted(names)
    return [
        _part(month, bbox, variables[offset : offset + 8], names, boundary=False)
        for offset in range(0, len(variables), 8)
    ] + [_accumulated_part(month, bbox)]


def _accumulated_part(month: date, bbox: tuple, *, boundary: bool = False) -> Part:
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values() if v.accumulated}
    names.update({EXTRA[name]: name for name in ('ssr', 'str')})
    return _part(month, bbox, sorted(names), names, boundary=boundary, accumulated=True)


def boundary_part(month: date, latest: date, bbox: tuple) -> Part:
    following = next_month(month)
    # Reuse the following month's midnights unless its full request is unavailable.
    incomplete = next_month(following) - const.ONE_DAY > latest
    return _accumulated_part(following, bbox, boundary=incomplete)


def needed_files(first: date, latest: date, bbox: tuple) -> set[str]:
    """Protect pending months and prefetched midnights during working-cache cleanup."""
    # The first unprocessed month may already have prefetched midnights even
    # when its own following midnight is not available yet.
    needed = {part.filename for part in month_parts(first, bbox)}
    month = first
    while next_month(month) <= latest:
        needed.update(
            part.filename
            for part in [*month_parts(month, bbox), boundary_part(month, latest, bbox)]
        )
        month = next_month(month)
    return needed


def _part(
    month: date,
    bbox: tuple,
    variables: list[str],
    names: dict[str, str],
    *,
    boundary: bool,
    accumulated: bool = False,
) -> Part:
    if boundary:
        day = ['01']
    else:
        day_count = calendar.monthrange(month.year, month.month)[1]
        day = [f'{d:02d}' for d in range(1, day_count + 1)]
    time_ = ['00:00'] if accumulated else [f'{h:02d}:00' for h in range(24)]
    request = {
        'variable': variables,
        'year': str(month.year),
        'month': f'{month.month:02d}',
        'day': day,
        'time': time_,
        'area': list(bbox),
        'data_format': 'grib',
        'download_format': 'unarchived',
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    return Part(month, boundary, request, {names[name] for name in variables}, key)
