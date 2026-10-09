from argparse import Namespace

from weatheasy import build_copdem
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    storage = Storage(args.output)
    build_copdem.build_copdem(
        tuple(args.bbox),
        args.resolution,
        storage.store(args.output),
    )
