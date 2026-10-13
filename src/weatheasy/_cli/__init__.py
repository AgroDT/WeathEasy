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
        help='weather Zarr store, local path or s3:// URI',
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        'weatheasy',
        description=(
            'Prepare and query weather data, correct biases, and interpolate station observations.'
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-v', '--version', action='version', version=version, help='show version and exit'
    )
    groups = parser.add_subparsers(required=True)
    groups.add_parser(
        'list-vars',
        help='list available weather variables',
        description='List weather variable names and descriptions for each source.',
    ).set_defaults(handler='list_vars')
    _add_query(groups)
    _add_download(groups)
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
    query = groups.add_parser(
        'query',
        help='query prepared weather data',
        description=(
            'Export daily weather variables for a point and an inclusive date range to CSV.'
        ),
    )
    sources = query.add_subparsers(dest='source', required=True)
    for source, variables in (
        ('cfs2', (*const.CFS2_BANDS, *const.CORRECTION_INFO)),
        ('cmip6', const.CMIP6_VARS),
        ('era5', (*const.ERA5_ALL_VARS, *const.CORRECTION_INFO)),
    ):
        sub = sources.add_parser(
            source,
            help=f'query prepared {source.upper()} weather data',
            description=f'Export daily {source.upper()} weather variables for a point to CSV.',
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(handler='query')
        _store(sub, source=source)
        sub.add_argument(
            'begin', type=date.fromisoformat, help='first date, inclusive (YYYY-MM-DD)'
        )
        sub.add_argument('end', type=date.fromisoformat, help='last date, inclusive (YYYY-MM-DD)')
        sub.add_argument(
            'latitude', type=float, help='latitude in decimal degrees WGS84 (EPSG:4326)'
        )
        sub.add_argument(
            'longitude', type=float, help='longitude in decimal degrees WGS84 (EPSG:4326)'
        )
        sub.add_argument(
            'variables',
            choices=variables,
            nargs='+',
            metavar='var',
            help='weather variable names; use list-vars to see available variables',
        )
        sub.add_argument(
            '-o', '--output', default='stdout', help='CSV output path, s3:// URI or stdout'
        )
        sub.add_argument(
            '-p',
            '--precision',
            type=int,
            choices=range(1, 7),
            default=6,
            help='number of decimal places in CSV values',
        )


def _add_download(groups: _SubParsersAction[ArgumentParser]) -> None:
    download = groups.add_parser(
        'download',
        help='download and prepare weather data',
        description=(
            'Download source weather files and write prepared data to a source-specific Zarr store.'
        ),
    )
    sources = download.add_subparsers(dest='source', required=True)
    for source in ('cfs2', 'cmip6', 'era5'):
        sub = sources.add_parser(
            source,
            help=f'download and prepare {source.upper()} weather data',
            description=(
                f'Download {source.upper()} weather data and update its prepared Zarr store.'
            ),
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(handler=f'download_{source}')
        _store(sub, source=source)
        sub.add_argument(
            '--download-cache',
            help='persistent raw-file cache, local path or s3:// URI; omitted uses temporary files',
        )
        sub.add_argument(
            '--work-dir',
            type=Path,
            help=(
                'local directory for remote cache files and CFS2 weather blocks; '
                'unset uses the platform state directory'
            ),
        )
        sub.add_argument(
            '--work-max-age-days',
            type=int,
            default=7,
            help='retention of unneeded working files in days; needed files are always retained',
        )
        if source == 'cfs2':
            sub.add_argument(
                '--artifacts',
                help='correction model directory or s3:// URI; '
                'when set, the downloaded forecast is bias-corrected (requires --dem)',
            )
            sub.add_argument('--dem', help='DEM Zarr path or s3:// URI for forecast correction')
        if source == 'era5':
            sub.add_argument(
                '--max-pending-requests',
                type=int,
                default=3,
                help='maximum outstanding CDS requests',
            )
            sub.add_argument(
                '--bbox',
                type=float,
                nargs=4,
                metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'),
                help=(
                    'bounding box in decimal degrees WGS84, without antimeridian '
                    'crossing; omitted covers the globe'
                ),
            )


def _dem_source(parser: ArgumentParser) -> None:
    parser.add_argument(
        '--dem-source',
        default='s3://datasets/copernicus-dem-90m.zarr',
        help=(
            'source elevation Zarr path, s3:// URI or HTTP(S) URL; '
            'requires EPSG:4326 (WGS84), assumed when CRS metadata is absent'
        ),
    )
    parser.add_argument(
        '--dem-elevation', default='elevation', help='source elevation variable in metres'
    )
    parser.add_argument('--dem-x', default='x', help='source longitude coordinate name')
    parser.add_argument('--dem-y', default='y', help='source latitude coordinate name')
    parser.add_argument(
        '--dem-headers-file',
        help=(
            'local JSON file of HTTP request headers, including authorization; HTTP(S) '
            'sources also support .netrc'
        ),
    )


def _add_correction(groups: _SubParsersAction[ArgumentParser]) -> None:
    correction = groups.add_parser(
        'correction',
        help='prepare terrain, train models, and apply bias correction',
        description=(
            'Prepare terrain and station features, train source-specific models, and '
            'correct daily weather.'
        ),
    )
    actions = correction.add_subparsers(required=True)
    prepare = actions.add_parser(
        'prepare-dem',
        help='prepare terrain on a weather source correction grid',
        description=(
            'Aggregate source DEM heights onto the weather correction grid and save '
            'elevation and slopes to a separate Zarr store for apply.'
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    prepare.set_defaults(handler='correction_prepare_dem')
    prepare.add_argument(
        'source', choices=('cfs2', 'era5'), help='weather source whose correction grid is used'
    )
    _store(prepare, required=True)
    prepare.add_argument('--output', '-o', required=True, help='terrain Zarr path or s3:// URI')
    _dem_source(prepare)
    prepare.add_argument('--overwrite', action='store_true', help='replace an existing output')
    stations = actions.add_parser(
        'prepare-stations',
        help='add native DEM features to station observations',
        description=(
            'Sample elevation and local slopes from the source DEM at station '
            'coordinates and write enriched observations for train.'
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    stations.set_defaults(handler='correction_prepare_stations')
    stations.add_argument(
        '--stations',
        required=True,
        help='station observations with lat/lon columns, Parquet path or s3:// URI',
    )
    stations.add_argument(
        '--output', '-o', required=True, help='enriched station Parquet path or s3:// URI'
    )
    _dem_source(stations)
    stations.add_argument('--overwrite', action='store_true', help='replace an existing output')
    for action in ('train', 'apply'):
        sub = actions.add_parser(
            action,
            help=(
                'train temperature and precipitation correction models'
                if action == 'train'
                else 'append corrected temperature and precipitation to the weather store'
            ),
            description=(
                'Train correction models from prepared weather and station '
                'observations enriched by prepare-stations.'
                if action == 'train'
                else (
                    'Apply trained models with terrain from prepare-dem and write '
                    'corrected fields to the weather store. Resume after the last '
                    'corrected day.'
                )
            ),
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(handler=f'correction_{action}')
        sub.add_argument('source', choices=('cfs2', 'era5'), help='weather source to correct')
        _store(sub, required=True)
        sub.add_argument(
            '--artifacts',
            default='artifacts',
            help=(
                'output model directory or s3:// URI'
                if action == 'train'
                else 'trained model directory or s3:// URI'
            ),
        )
        if action == 'train':
            sub.add_argument('--dem', help='prepared terrain Zarr for forecast quality report')
            sub.add_argument(
                '--stations',
                required=True,
                help='station Parquet with prepared terrain features, local path or s3:// URI',
            )
            sub.add_argument(
                '--only',
                choices=('temp', 'precip'),
                help='train only the selected model; omitted trains both',
            )
            sub.add_argument(
                '--cv-folds',
                type=int,
                default=5,
                help='station-grouped folds for the quality report; 0 skips the scores',
            )
        else:
            sub.add_argument(
                '--dem',
                required=True,
                help='terrain from prepare-dem for this weather source, Zarr path or s3:// URI',
            )
            sub.add_argument(
                '--begin',
                type=date.fromisoformat,
                help=(
                    'first date, inclusive (YYYY-MM-DD); omitted starts from the first pending day'
                ),
            )
            sub.add_argument(
                '--end',
                type=date.fromisoformat,
                help=(
                    'last date, inclusive (YYYY-MM-DD); omitted includes all committed weather days'
                ),
            )


def _add_emdna(groups: _SubParsersAction[ArgumentParser]) -> None:
    emdna = groups.add_parser(
        'emdna',
        help='interpolate station observations',
        description=(
            'Interpolate daily station observations and generate spatially correlated ensembles.'
        ),
    )
    kinds = emdna.add_subparsers(dest='kind', required=True)
    for kind in ('temperature', 'precipitation'):
        sub = kinds.add_parser(
            kind,
            help=f'interpolate {kind} and generate ensembles',
            description=(
                f'Interpolate daily {kind} observations and save ensembles, '
                f'uncertainty, and restart state.'
            ),
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(handler='emdna')
        sub.add_argument('stations', help='station Parquet path or s3:// URI')
        sub.add_argument(
            'begin',
            nargs='?',
            type=date.fromisoformat,
            help='first date, inclusive (YYYY-MM-DD); omitted uses remaining observation dates',
        )
        sub.add_argument(
            'end',
            nargs='?',
            type=date.fromisoformat,
            help='last date, inclusive (YYYY-MM-DD); omitted includes all observation dates',
        )
        sub.add_argument(
            '-o', '--output', default=f'emdna_{kind}_results', help='result directory or s3:// URI'
        )
        sub.add_argument(
            '--era5',
            help=(
                'optional ERA5 predictors: directory of monthly daily NetCDF files, '
                'local or s3:// URI'
            ),
        )
        sub.add_argument('--n-ensemble', type=int, default=100, help='number of ensemble members')
        sub.add_argument(
            '--min-stations',
            type=int,
            default=20,
            help=(
                'minimum stations required per day and sought for each cell, extending '
                'beyond the search radius if needed'
            ),
        )
        sub.add_argument(
            '--max-stations',
            type=int,
            default=30,
            help='maximum nearest stations selected per cell',
        )
        sub.add_argument(
            '--search-radius', type=float, default=400.0, help='station search radius in km'
        )
