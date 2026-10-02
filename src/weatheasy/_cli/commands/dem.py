from argparse import Namespace

from weatheasy.build_copdem import build_copdem
from weatheasy.storage import get_store


def run(args: Namespace) -> None:
    build_copdem(tuple(args.area), args.resolution, get_store(args.output), args.cache_dir)
