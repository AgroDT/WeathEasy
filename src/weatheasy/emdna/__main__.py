from __future__ import annotations

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._base import EMDNABase


def main(*, configure_logging: bool = True) -> None:
    if configure_logging:
        logging.basicConfig(
            format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
            level=logging.INFO,
        )

    parser = ArgumentParser(
        prog='weatheasy.emdna',
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='EMDNA spatial interpolation with ensemble generation',
    )
    subparsers = parser.add_subparsers(dest='kind', required=True)

    for name in ('temperature', 'precipitation'):
        sub = subparsers.add_parser(name, formatter_class=ArgumentDefaultsHelpFormatter)
        sub.add_argument(
            'stations', type=Path, metavar='STATIONS', help='CSV file with station observations'
        )
        sub.add_argument(
            'begin',
            nargs='?',
            type=date.fromisoformat,
            metavar='BEGIN',
            help='first date yyyy-mm-dd',
        )
        sub.add_argument(
            'end', nargs='?', type=date.fromisoformat, metavar='END', help='last date yyyy-mm-dd'
        )
        sub.add_argument(
            '-o',
            '--output',
            type=Path,
            default=Path.cwd() / f'emdna_{name}_results',
            metavar='DIR',
            help='output directory for .npz files',
        )
        sub.add_argument(
            '--era5',
            type=Path,
            metavar='DIR',
            help='directory with ERA5-Land daily NetCDF files (optional)',
        )
        sub.add_argument(
            '--n-ensemble', type=int, default=100, metavar='N', help='number of ensemble members'
        )
        sub.add_argument(
            '--min-stations',
            type=int,
            default=20,
            metavar='N',
            help='minimum number of stations per grid point',
        )
        sub.add_argument(
            '--max-stations',
            type=int,
            default=30,
            metavar='N',
            help='maximum number of stations per grid point',
        )
        sub.add_argument(
            '--search-radius',
            type=float,
            default=400.0,
            metavar='KM',
            help='station search radius in km',
        )

    args = parser.parse_args()

    processor: EMDNABase
    try:
        if args.kind == 'temperature':
            from .temperature import TemperatureProcessor

            processor = TemperatureProcessor(
                station_file=args.stations,
                output_dir=args.output,
                era5_dir=args.era5,
                n_ensemble=args.n_ensemble,
                min_stations=args.min_stations,
                max_stations=args.max_stations,
                search_radius_km=args.search_radius,
            )
        else:
            from .precipitation import PrecipitationProcessor

            processor = PrecipitationProcessor(
                station_file=args.stations,
                output_dir=args.output,
                era5_dir=args.era5,
                n_ensemble=args.n_ensemble,
                min_stations=args.min_stations,
                max_stations=args.max_stations,
                search_radius_km=args.search_radius,
            )
    except ImportError:
        logging.exception('EMDNA requires scipy and xarray: pip install weatheasy[emdna]')
        sys.exit(1)

    results = processor.process_period(args.begin, args.end)

    n_ok = sum(1 for r in results if r['status'] in ('success', 'exists'))
    n_err = sum(1 for r in results if r['status'] not in ('success', 'exists'))
    logging.info('Done: %d/%d days processed, %d errors', n_ok, len(results), n_err)

    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
