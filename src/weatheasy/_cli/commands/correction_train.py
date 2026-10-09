from argparse import Namespace

from weatheasy._cli._inputs import read_stations
from weatheasy._cli._storage import Storage
from weatheasy.correction import training
from weatheasy.correction.sources import SOURCES


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.dem, args.artifacts, args.stations)
    weather = storage.group(args.store)
    dem = storage.store(args.dem)
    artifacts = storage.files(args.artifacts)
    source = SOURCES[args.source]
    columns = ['station_id', 'lat', 'lon', 'date']
    if args.only in (None, 'temp'):
        columns.append('temperature')
    if args.only in (None, 'precip'):
        columns.append('precipitation')
    stations = read_stations(storage.files(args.stations), columns=columns)
    training.build_artifacts(source, weather, stations, dem, artifacts, only=args.only)
