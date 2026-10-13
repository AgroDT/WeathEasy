from argparse import Namespace

from weatheasy._cli._inputs import read_stations
from weatheasy._cli._storage import Storage
from weatheasy.correction import training
from weatheasy.correction.sources import SOURCES


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.artifacts, args.stations)
    weather = storage.group(args.store)
    artifacts = storage.files(args.artifacts)
    source = SOURCES[args.source]
    columns = ['station_id', 'lat', 'lon', 'date', 'elevation', 'slope_we', 'slope_sn']
    if args.only in (None, 'temp'):
        columns.append('temperature')
    if args.only in (None, 'precip'):
        columns.append('precipitation')
    stations = read_stations(storage.files(args.stations), columns=columns)
    training.build_artifacts(source, weather, stations, artifacts, only=args.only)
