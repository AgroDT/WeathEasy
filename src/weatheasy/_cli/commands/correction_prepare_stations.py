from argparse import Namespace

from weatheasy._cli._inputs import read_stations
from weatheasy._cli._storage import Storage
from weatheasy.correction.terrain import prepare_station_terrain


def run(args: Namespace) -> None:
    storage = Storage(args.stations, args.dem_source, args.output)
    output = storage.files(args.output)
    if output.exists() and not args.overwrite:
        msg = f'Output already exists: {args.output}; use --overwrite to replace it'
        raise FileExistsError(msg)
    observations = read_stations(storage.files(args.stations))
    prepared = prepare_station_terrain(
        observations,
        storage.dem_source(args.dem_source, headers_file=args.dem_headers_file),
        elevation=args.dem_elevation,
        x=args.dem_x,
        y=args.dem_y,
    )
    # FileStore publishes a complete Parquet file using the existing local/S3 path.
    with output.write_atomic('') as stream:
        prepared.to_parquet(stream, engine='pyarrow', index=False)
