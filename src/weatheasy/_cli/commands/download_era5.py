from argparse import Namespace

from weatheasy.era5 import download_era5_data
from weatheasy.storage import get_store


def run(args: Namespace) -> None:
    download_era5_data(
        get_store(args.store), args.download_dir, area=tuple(args.area) if args.area else None
    )
