from argparse import Namespace

from weatheasy import download
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.download_cache)
    cache = storage.files(args.download_cache) if args.download_cache else None
    download.download_cmip6_data(
        storage.store(args.store),
        cache,
        work_dir=args.work_dir,
        work_max_age_days=args.work_max_age_days,
    )
