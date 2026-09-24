"""One-shot training and application jobs for reanalysis correction."""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import date
from pathlib import Path

from .sources import SOURCES
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
    parser.add_argument('--dem', type=Path, required=True, help='Copernicus DEM NetCDF')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    if action == 'train':
        parser.add_argument('--stations', type=Path, required=True)
        parser.add_argument('--only', choices=('temp', 'precip'))
    else:
        parser.add_argument('--begin', type=date.fromisoformat)
        parser.add_argument('--end', type=date.fromisoformat)
    args = parser.parse_args(args_list)
    root = get_storage(args.store)
    source = SOURCES[args.source]
    if action == 'train':
        from .training import build_artifacts

        build_artifacts(source, root, args.stations, args.dem, args.artifacts, only=args.only)
    else:
        from .apply import correct_store

        days = correct_store(
            source,
            root,
            args.artifacts,
            args.dem,
            begin=args.begin,
            end=args.end,
        )
        logging.getLogger(__name__).info('Corrected %d days for %s', days, source.name)


if __name__ == '__main__':
    main()
