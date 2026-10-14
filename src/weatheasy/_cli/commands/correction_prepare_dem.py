from argparse import Namespace

from weatheasy._cli._storage import Storage
from weatheasy.correction.sources import SOURCES
from weatheasy.correction.terrain import prepare_dem


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.dem_source, args.output)
    prepare_dem(
        SOURCES[args.source],
        storage.group(args.store),
        storage.dem_source(args.dem_source, headers_file=args.dem_headers_file),
        storage.store(args.output),
        elevation=args.dem_elevation,
        x=args.dem_x,
        y=args.dem_y,
        overwrite=args.overwrite,
    )
