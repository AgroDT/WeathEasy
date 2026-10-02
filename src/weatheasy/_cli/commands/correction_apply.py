import logging
from argparse import Namespace

from weatheasy._cli._storage import Storage
from weatheasy.correction import apply
from weatheasy.correction.sources import SOURCES


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.dem, args.artifacts)
    weather = storage.group(args.store)
    dem = storage.store(args.dem)
    artifacts = storage.files(args.artifacts)
    source = SOURCES[args.source]
    days = apply.correct_store(source, weather, artifacts, dem, begin=args.begin, end=args.end)
    logging.getLogger(__name__).info('Corrected %d days for %s', days, source.name)
