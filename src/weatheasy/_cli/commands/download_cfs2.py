import logging
from argparse import Namespace

from weatheasy import download
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    if (args.artifacts is None) != (args.dem is None):
        msg = 'Forecast correction requires both --artifacts and --dem'
        raise ValueError(msg)
    storage = Storage(args.store, args.download_cache, args.dem, args.artifacts)
    cache = storage.files(args.download_cache) if args.download_cache else None
    download.download_cfs2_data(
        storage.store(args.store),
        cache,
        work_dir=args.work_dir,
        work_max_age_days=args.work_max_age_days,
    )
    if args.artifacts is None:
        return
    # Imported here: plain downloads do not need the correction extra.
    from weatheasy.correction import apply  # noqa: PLC0415

    days = apply.correct_forecast(
        storage.group(args.store), storage.files(args.artifacts), storage.store(args.dem)
    )
    logging.getLogger(__name__).info('Corrected %d forecast days', days)
