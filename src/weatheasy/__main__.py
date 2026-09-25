from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from weatheasy import Coords, const, get_cfs2_data, get_cmip6_data, get_era5_data
from weatheasy.storage import get_store
from weatheasy.util import float_formatter_factory, init_parser


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction
    from collections.abc import Iterable


_CORRECTION_INFO = {
    'TMP_corr': const.VarInfo(
        en='Bias-corrected daily temperature, °C',
        ru='Скорректированная температура, °C',
    ),
    'PRATE_corr': const.VarInfo(
        en='Bias-corrected daily precipitation, mm/day',
        ru='Скорректированные осадки, мм/сутки',
    ),
}


def main(*, configure_logging: bool = True) -> None:
    if configure_logging:
        logging.basicConfig(
            format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
            level=logging.WARNING,
        )
    args = _parse_args()
    if args.action == 'list-vars':
        _print_vars('ERA5-Land variables:', const.ERA5_ALL_VARS.items())
        _print_vars('Corrected ERA5 variables:', _CORRECTION_INFO.items())
        _print_vars('CMIP6 variables:', const.CMIP6_VARS.items())
        _print_vars('CFS2 variables:', ((k, v.info) for k, v in const.CFS2_BANDS.items()))
    else:
        _run(args)


def _parse_args():
    parser = init_parser()
    parser.add_argument(
        '-o',
        '--output',
        metavar='PATH',
        help='output file',
        default='stdout',
    )
    parser.add_argument(
        '-p',
        '--precision',
        metavar='INT',
        help='Number of decimal places for rounding results in responses',
        default=6,
    )
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('list-vars', help='list available variables')
    _add_data_subparser(subparsers, 'era5', (*const.ERA5_ALL_VARS, *_CORRECTION_INFO))
    _add_data_subparser(subparsers, 'cmip6', const.CMIP6_VARS)
    _add_data_subparser(subparsers, 'cfs2', (*const.CFS2_BANDS, *_CORRECTION_INFO))

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

    return parser


def _run(args: Namespace):
    if args.action == 'cfs2':
        func = get_cfs2_data
    elif args.action == 'era5':
        func = get_era5_data
    elif args.action == 'cmip6':
        func = get_cmip6_data
    else:
        raise RuntimeError

    format_float = float_formatter_factory('NA', args.precision)
    store = get_store(args.data)
    begin: date = args.begin
    end: date = args.end
    coords = Coords(latitude=args.latitude, longitude=args.longitude)
    variables: list[str] = args.variables
    data = func(
        store=store,
        begin=begin,
        end=end,
        coords=coords,
        variables=variables,
    )

    if args.output == 'stdout':
        output = sys.stdout
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open('w')

    with output as f:
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
                f.write(format_float(cell))
            f.write('\n')


def _print_vars(header: str, variables: Iterable[tuple[str, const.VarInfo]]) -> None:
    print(header)  # noqa: T201
    for var, info in variables:
        print(f'  {var:<16}{info.en}')  # noqa: T201
        if ru := info.ru:
            print(f'                  {ru}')  # noqa: T201
    print()  # noqa: T201


if __name__ == '__main__':
    main()
