from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from weatheasy._cli._inputs import process_period, read_stations
from weatheasy._cli._storage import Storage


def run(args: Namespace) -> None:
    storage = Storage(args.stations, args.output, args.era5)
    try:
        module = importlib.import_module(f'weatheasy.emdna.{args.kind}')
    except ImportError:
        logging.getLogger(__name__).exception(
            'EMDNA requires scipy and xarray: pip install weatheasy[emdna]'
        )
        raise SystemExit(1) from None
    processor_class = (
        module.TemperatureProcessor if args.kind == 'temperature' else module.PrecipitationProcessor
    )
    processor = processor_class(
        stations=read_stations(storage.files(args.stations)),
        output_dir=storage.files(args.output),
        n_ensemble=args.n_ensemble,
        min_stations=args.min_stations,
        max_stations=args.max_stations,
        search_radius_km=args.search_radius,
    )
    results = process_period(
        processor, storage.files(args.era5) if args.era5 else None, args.begin, args.end
    )
    n_ok = sum(result['status'] in ('success', 'exists') for result in results)
    n_err = len(results) - n_ok
    logging.getLogger(__name__).info(
        'Done: %d/%d days processed, %d errors', n_ok, len(results), n_err
    )
    if n_err:
        raise SystemExit(1)


if TYPE_CHECKING:
    from argparse import Namespace
