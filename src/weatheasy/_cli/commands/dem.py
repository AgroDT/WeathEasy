from argparse import Namespace

from weatheasy import build_copdem
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    storage = Storage(args.output, args.download_cache)
    build_copdem.build_copdem(
        tuple(args.bbox),
        args.resolution,
        storage.store(args.output),
        storage.files(args.download_cache),
    )
