from argparse import Namespace

from weatheasy.download import download_cmip6_data
from weatheasy.storage import get_store


def run(args: Namespace) -> None:
    download_cmip6_data(get_store(args.store), args.download_dir)
