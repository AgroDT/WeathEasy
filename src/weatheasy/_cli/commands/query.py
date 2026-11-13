from __future__ import annotations

import sys
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from weatheasy import const
from weatheasy._cli._storage import Storage
from weatheasy.file_storage import FileStore
from weatheasy.query import Coords, get_cfs2_data, get_cmip6_data, get_era5_data
from weatheasy.util import float_formatter_factory


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

    import numpy as np


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.output)
    getter = {'cfs2': get_cfs2_data, 'cmip6': get_cmip6_data, 'era5': get_era5_data}[args.source]
    data = getter(
        store=storage.store(args.store),
        begin=args.begin,
        end=args.end,
        coords=Coords(args.latitude, args.longitude),
        variables=args.variables,
    )
    if args.output == 'stdout':
        _write_csv(args, data, sys.stdout.write)
    else:
        target = storage.files(args.output)
        path = PurePosixPath(target.root)
        with FileStore(target.fs, str(path.parent)).write_atomic(path.name) as stream:
            _write_csv(args, data, lambda text: stream.write(text.encode('utf-8')))


def _write_csv(args: Namespace, data: np.ndarray, write: Callable[[str], object]) -> None:
    format_float = float_formatter_factory('NA', args.precision)
    write('DATE,' + ','.join(args.variables) + '\n')
    day = args.begin
    for row in data.transpose():
        write(f'{day:%Y-%m-%d},' + ','.join(format_float(cell) for cell in row) + '\n')
        day += const.ONE_DAY
