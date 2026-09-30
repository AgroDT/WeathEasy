from __future__ import annotations

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import date
from typing import TYPE_CHECKING

from weatheasy.file_storage import get_file_store


if TYPE_CHECKING:
    from ._base import EMDNABase


_LOG = logging.getLogger(__name__)


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
        sub.add_argument('stations', metavar='STATIONS', help='CSV file with station observations')
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
            default=f'emdna_{name}_results',
            metavar='DIR',
            help='output directory for .npz files',
        )
        sub.add_argument(
            '--era5',
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

    locations = [args.stations, args.output, args.era5]
    s3_uri = next((value for value in locations if value and value.startswith('s3://')), None)
    filesystem = get_file_store(s3_uri).fs if s3_uri is not None else None
    station_file = get_file_store(
        args.stations, filesystem=filesystem if args.stations.startswith('s3://') else None
    )
    output_dir = get_file_store(
        args.output, filesystem=filesystem if args.output.startswith('s3://') else None
    )
    era5_dir = (
        get_file_store(args.era5, filesystem=filesystem if args.era5.startswith('s3://') else None)
        if args.era5 is not None
        else None
    )

    processor: EMDNABase
    try:
        from ._inputs import process_period, read_stations  # noqa: PLC0415

        observations = read_stations(station_file)
        if args.kind == 'temperature':
            # Keep the selected processor import inside the CLI error handler.
            from .temperature import TemperatureProcessor  # noqa: PLC0415

            processor = TemperatureProcessor(
                stations=observations,
                output_dir=output_dir,
                n_ensemble=args.n_ensemble,
                min_stations=args.min_stations,
                max_stations=args.max_stations,
                search_radius_km=args.search_radius,
            )
        else:
            # Keep the selected processor import inside the CLI error handler.
            from .precipitation import PrecipitationProcessor  # noqa: PLC0415

            processor = PrecipitationProcessor(
                stations=observations,
                output_dir=output_dir,
                n_ensemble=args.n_ensemble,
                min_stations=args.min_stations,
                max_stations=args.max_stations,
                search_radius_km=args.search_radius,
            )
    except ImportError:
        _LOG.exception('EMDNA requires scipy and xarray: pip install weatheasy[emdna]')
        sys.exit(1)

    results = process_period(processor, era5_dir, args.begin, args.end)

    n_ok = sum(1 for r in results if r['status'] in ('success', 'exists'))
    n_err = sum(1 for r in results if r['status'] not in ('success', 'exists'))
    _LOG.info('Done: %d/%d days processed, %d errors', n_ok, len(results), n_err)

    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
