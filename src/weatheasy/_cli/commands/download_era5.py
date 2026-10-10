from argparse import Namespace

from weatheasy import era5
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    storage = Storage(args.store, args.download_cache)
    cache = storage.files(args.download_cache) if args.download_cache else None
    era5.download_era5_data(
        storage.store(args.store),
        cache,
        work_dir=args.work_dir,
        work_max_age_days=args.work_max_age_days,
        bbox=tuple(args.bbox) if args.bbox else None,
        max_pending_requests=args.max_pending_requests,
    )
