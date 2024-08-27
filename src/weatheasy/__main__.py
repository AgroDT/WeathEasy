from __future__ import annotations

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace, _SubParsersAction
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from weatheasy import Coords, const, get_cfs2_data, get_cmip6_data
from weatheasy.util import get_storage


if TYPE_CHECKING:
    from collections.abc import Iterable


def main(*, configure_logging: bool = True) -> None:
    if configure_logging:
        logging.basicConfig(
            format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
            level=logging.INFO,
        )
    args = _parse_args()
    if args.action == 'list-vars':
        _print_vars('CMIP6 variables:', const.CMIP6_VARS.items())
        _print_vars('CFS2 variables:', ((k, v.info) for k, v in const.CFS2_BANDS.items()))
    else:
        _run(args)


def _parse_args():
    parser = ArgumentParser(__package__, formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('list-vars', help='list available variables')
    _add_data_subparser(subparsers, 'cmip6', const.CMIP6_VARS)
    _add_data_subparser(subparsers, 'cfs2', const.CFS2_BANDS)

    return parser.parse_args()


def _add_data_subparser(
    subparsers: _SubParsersAction[ArgumentParser],
    name: str,
    variables: Iterable[str],
):
    parser = subparsers.add_parser(name)
    parser.add_argument('begin', type=date.fromisoformat, help='first date yyyy-mm-dd')
    parser.add_argument('end', type=date.fromisoformat, help='last date yyyy-mm-dd')
    parser.add_argument('latitude', type=float, help='decimal degrees EPSG:4326')
    parser.add_argument('longitude', type=float, help='decimal degrees EPSG:4326')
    parser.add_argument(
        'variables',
        help='output variables (run `weatheasy list-vars` for a full list of variables)',
        choices=variables,
        nargs='+',
        metavar='var',
    )
    parser.add_argument(
        '-d',
        '--data',
        default=Path.cwd().as_posix(),
        metavar='STORE',
        help='Zarr store for downloaded data',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        metavar='PATH',
        help='output file',
    )
    return parser


def _run(args: Namespace):
    if args.action == 'cfs2':
        func = get_cfs2_data
    elif args.action == 'cmip6':
        func = get_cmip6_data
    else:
        raise RuntimeError

    root = get_storage(args.data)
    begin: date = args.begin
    end: date = args.end
    coords = Coords(latitude=args.latitude, longitude=args.longitude)
    variables: list[str] = args.variables
    data = func(
        root=root,
        begin=begin,
        end=end,
        coords=coords,
        variables=variables,
    )

    output: Path | None = args.output
    if output is None:
        lat = _format_deg(coords.latitude, 'S', 'N')
        lon = _format_deg(coords.longitude, 'W', 'E')
        output = Path.cwd() / f'{lat}{lon}_{begin:%Y%m%d}-{end:%Y%m%d}.csv'
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        f.write('DATE')
        for cell in variables:
            f.write(',')
            f.write(cell)
        f.write('\n')
        date_ = begin
        for row in data.transpose():
            f.write(f'{date_:%Y-%m-%d}')
            date_ += const.ONE_DAY
            for cell in row:
                f.write(',')
                f.write('NA' if np.isnan(cell) else f'{cell:.6f}')
            f.write('\n')


def _print_vars(header: str, variables: Iterable[tuple[str, const.VarInfo]]) -> None:
    print(header)  # noqa: T201
    for var, info in variables:
        print(f'  {var:<16}{info.en}')  # noqa: T201
        if ru := info.ru:
            print(f'                  {ru}')  # noqa: T201
    print()  # noqa: T201


def _format_deg(deg: float, lower: str, upper: str) -> str:
    if deg >= 0:
        hem = upper
    else:
        deg *= -1
        hem = lower
    return f'{hem}{deg * 100000:0>8.0f}'


if __name__ == '__main__':
    main()
