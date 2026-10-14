from __future__ import annotations

from typing import TYPE_CHECKING

from weatheasy import const


def _print_vars(header: str, variables: Iterable[tuple[str, const.VarInfo]]) -> None:
    print(header)  # noqa: T201
    for var, info in variables:
        print(f'  {var:<16}{info.en}')  # noqa: T201
        if ru := info.ru:
            print(f'                  {ru}')  # noqa: T201
    print()  # noqa: T201


def run(_args: Namespace) -> None:
    _print_vars('ERA5-Land variables:', const.ERA5_ALL_VARS.items())
    _print_vars('Corrected ERA5 variables:', const.CORRECTION_INFO.items())
    _print_vars('CMIP6 variables:', const.CMIP6_VARS.items())
    _print_vars('CFS2 variables:', ((k, v.info) for k, v in const.CFS2_BANDS.items()))


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterable
