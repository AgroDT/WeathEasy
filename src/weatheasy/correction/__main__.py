"""One-shot training and application jobs for reanalysis correction."""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import date

from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

from .sources import SOURCES
from weatheasy.file_storage import get_file_store
from weatheasy.storage import get_store
from weatheasy.util import get_storage


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        level=logging.INFO,
    )
    args_list = list(sys.argv[1:] if argv is None else argv)
    action = 'apply' if args_list and args_list[0] == 'apply' else 'train'
    if args_list and args_list[0] in ('apply', 'train'):
        args_list.pop(0)
    parser = ArgumentParser(
        f'weatheasy.correction {action}',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('source', choices=sorted(SOURCES))
    parser.add_argument('--store', required=True, help='prepared Zarr v3 store')
    parser.add_argument('--dem', required=True, help='Copernicus DEM Zarr path or s3:// URI')
    parser.add_argument('--artifacts', default='artifacts', help='model directory or s3:// URI')
    if action == 'train':
        parser.add_argument('--stations', required=True, help='station CSV path or s3:// URI')
        parser.add_argument('--only', choices=('temp', 'precip'))
    else:
        parser.add_argument('--begin', type=date.fromisoformat)
        parser.add_argument('--end', type=date.fromisoformat)
    args = parser.parse_args(args_list)
    locations = [args.store, args.dem, args.artifacts]
    if action == 'train':
        locations.append(args.stations)
    s3_uri = next((uri for uri in locations if uri.startswith('s3://')), None)
    filesystem = get_file_store(s3_uri).fs if s3_uri is not None else None
    # Wrap the synchronous client for Zarr, keeping the same filesystem for model streams.
    async_fs = (
        AsyncFileSystemWrapper(fs=filesystem, asynchronous=True) if filesystem is not None else None
    )
    root = get_storage(args.store, filesystem=async_fs if args.store.startswith('s3://') else None)
    dem_store = get_store(args.dem, filesystem=async_fs if args.dem.startswith('s3://') else None)
    artifacts = get_file_store(
        args.artifacts, filesystem=filesystem if args.artifacts.startswith('s3://') else None
    )
    source = SOURCES[args.source]
    if action == 'train':
        import pandas as pd  # noqa: PLC0415

        # Load the training pipeline only for the train command.
        from .training import build_artifacts  # noqa: PLC0415

        station_file = get_file_store(
            args.stations, filesystem=filesystem if args.stations.startswith('s3://') else None
        )
        with station_file.open() as stream:
            stations = pd.read_csv(stream)
        build_artifacts(source, root, stations, dem_store, artifacts, only=args.only)
    else:
        # Load the application pipeline only for the apply command.
        from .apply import correct_store  # noqa: PLC0415

        days = correct_store(
            source,
            root,
            artifacts,
            dem_store,
            begin=args.begin,
            end=args.end,
        )
        logging.getLogger(__name__).info('Corrected %d days for %s', days, source.name)


if __name__ == '__main__':
    main()
