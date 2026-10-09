"""Command metadata and lazy command dispatch."""

from __future__ import annotations

import importlib
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, _SubParsersAction
from datetime import date
from pathlib import Path

from weatheasy import const


try:
    from weatheasy._version import version
except ImportError:
    version = 'unknown version'


def _store(parser: ArgumentParser, *, source: str = '', required: bool = False) -> None:
    parser.add_argument(
        '--store',
        '-d',
        '--data',
        required=required,
        default=None if required else str(Path.cwd() / 'weatheasy' / f'{source}.zarr'),
        help='prepared source-specific Zarr path or s3:// URI',
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser('weatheasy', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=version)
    groups = parser.add_subparsers(required=True)
    groups.add_parser('list-vars', help='list available variables').set_defaults(
        handler='list_vars'
    )
    _add_query(groups)
    _add_download(groups)
    _add_dem(groups)
    _add_correction(groups)
    _add_emdna(groups)
    return parser


def main(argv: list[str] | None = None, *, configure_logging: bool = True) -> None:
    args = _build_parser().parse_args(argv)
    if configure_logging:
        logging.basicConfig(
            format='%(asctime)s %(name)s [%(levelname)s] %(message)s', level=logging.INFO
        )
    command = importlib.import_module(f'weatheasy._cli.commands.{args.handler}')
    command.run(args)


def _add_query(groups: _SubParsersAction[ArgumentParser]) -> None:
    query = groups.add_parser('query', help='query prepared weather data')
    sources = query.add_subparsers(dest='source', required=True)
    for source, variables in (
        ('cfs2', (*const.CFS2_BANDS, *const.CORRECTION_INFO)),
        ('cmip6', const.CMIP6_VARS),
        ('era5', (*const.ERA5_ALL_VARS, *const.CORRECTION_INFO)),
    ):
        sub = sources.add_parser(source, formatter_class=ArgumentDefaultsHelpFormatter)
        sub.set_defaults(handler='query')
        _store(sub, source=source)
        sub.add_argument('begin', type=date.fromisoformat, help='first date yyyy-mm-dd')
        sub.add_argument('end', type=date.fromisoformat, help='last date yyyy-mm-dd')
        sub.add_argument('latitude', type=float, help='decimal degrees EPSG:4326')
        sub.add_argument('longitude', type=float, help='decimal degrees EPSG:4326')
        sub.add_argument('variables', choices=variables, nargs='+', metavar='var')
        sub.add_argument(
            '-o', '--output', default='stdout', help='CSV output path, s3:// URI or stdout'
        )
        sub.add_argument('-p', '--precision', type=int, choices=range(1, 7), default=6)


def _add_download(groups: _SubParsersAction[ArgumentParser]) -> None:
    download = groups.add_parser('download', help='download and prepare weather data')
    sources = download.add_subparsers(dest='source', required=True)
    for source in ('cfs2', 'cmip6', 'era5'):
        sub = sources.add_parser(source, formatter_class=ArgumentDefaultsHelpFormatter)
        sub.set_defaults(handler=f'download_{source}')
        _store(sub, source=source)
        sub.add_argument(
            '--download-cache', help='cache for downloaded files, local path or s3:// URI'
        )
        if source == 'era5':
            sub.add_argument(
                '--bbox',
                type=float,
                nargs=4,
                metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'),
                help='bounding box in decimal degrees EPSG:4326, without antimeridian crossing',
            )


def _add_dem(groups: _SubParsersAction[ArgumentParser]) -> None:
    dem = groups.add_parser(
        'dem', help='prepare terrain features', formatter_class=ArgumentDefaultsHelpFormatter
    )
    dem.set_defaults(handler='dem')
    dem.add_argument(
        '--bbox',
        type=int,
        nargs=4,
        required=True,
        metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'),
        help='bounding box in whole degrees EPSG:4326, without antimeridian crossing',
    )
    dem.add_argument('--resolution', type=float, default=0.1)
    dem.add_argument('-o', '--output', required=True, help='Zarr path or s3:// URI')


def _add_correction(groups: _SubParsersAction[ArgumentParser]) -> None:
    correction = groups.add_parser('correction', help='train and apply bias correction')
    actions = correction.add_subparsers(required=True)
    for action in ('train', 'apply'):
        sub = actions.add_parser(action, formatter_class=ArgumentDefaultsHelpFormatter)
        sub.set_defaults(handler=f'correction_{action}')
        sub.add_argument('source', choices=('cfs2', 'era5'))
        _store(sub, required=True)
        sub.add_argument('--dem', required=True, help='DEM Zarr path or s3:// URI')
        sub.add_argument('--artifacts', default='artifacts', help='model directory or s3:// URI')
        if action == 'train':
            sub.add_argument('--stations', required=True, help='station Parquet path or s3:// URI')
            sub.add_argument('--only', choices=('temp', 'precip'))
        else:
            sub.add_argument('--begin', type=date.fromisoformat)
            sub.add_argument('--end', type=date.fromisoformat)


def _add_emdna(groups: _SubParsersAction[ArgumentParser]) -> None:
    emdna = groups.add_parser('emdna', help='interpolate station observations')
    kinds = emdna.add_subparsers(dest='kind', required=True)
    for kind in ('temperature', 'precipitation'):
        sub = kinds.add_parser(kind, formatter_class=ArgumentDefaultsHelpFormatter)
        sub.set_defaults(handler='emdna')
        sub.add_argument('stations', help='station Parquet path or s3:// URI')
        sub.add_argument('begin', nargs='?', type=date.fromisoformat)
        sub.add_argument('end', nargs='?', type=date.fromisoformat)
        sub.add_argument(
            '-o', '--output', default=f'emdna_{kind}_results', help='result directory or s3:// URI'
        )
        sub.add_argument('--era5', help='directory with monthly daily NetCDF files, local or s3://')
        sub.add_argument('--n-ensemble', type=int, default=100)
        sub.add_argument('--min-stations', type=int, default=20)
        sub.add_argument('--max-stations', type=int, default=30)
        sub.add_argument(
            '--search-radius', type=float, default=400.0, help='station search radius in km'
        )
