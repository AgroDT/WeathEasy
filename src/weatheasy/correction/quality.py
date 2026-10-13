"""Quality report written when correction models are trained.

Reanalysis skill is measured against station observations held out by
station-grouped cross-validation. The forecast has no matching observations yet,
so its section only shows how the reanalysis models shift the current forecast,
next to the shift they apply to the reanalysis in the same calendar months.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .apply import open_regridded
from .precip import WET_THRESHOLD, PrecipCorrector
from .sources import SOURCES
from .temperature import TemperatureCorrector
from weatheasy.storage import open_dataset


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    import zarr
    from numpy.typing import NDArray
    from zarr.abc.store import Store

    from .sources import Source
    from weatheasy.file_storage import FileStore


_LOG = logging.getLogger(__name__)

REPORT_JSON = 'quality_report.json'
REPORT_MARKDOWN = 'quality_report.md'
_SAMPLE_STEP = 7  # every n-th forecast day enters the consistency check
_REANALYSIS_DAYS_PER_MONTH = 4

_TITLES = {'era5': 'ERA5-Land', 'cfs2': 'CFSv2'}


# ── Metrics ─────────────────────────────────────────────────────────────────


def _continuous(obs: NDArray, values: NDArray) -> dict[str, float]:
    diff = values - obs
    corr = (
        float(np.corrcoef(obs, values)[0, 1])
        if len(obs) > 1 and np.std(obs) > 0 and np.std(values) > 0
        else float('nan')
    )
    return {
        'bias': float(diff.mean()),
        'mae': float(np.abs(diff).mean()),
        'rmse': float(np.sqrt((diff**2).mean())),
        'corr': corr,
    }


def _wet_scores(obs: NDArray, values: NDArray) -> dict[str, float]:
    obs_wet = obs >= WET_THRESHOLD
    wet = values >= WET_THRESHOLD
    hits = float((wet & obs_wet).sum())
    misses = float((~wet & obs_wet).sum())
    false_alarms = float((wet & ~obs_wet).sum())
    negatives = float((~wet & ~obs_wet).sum())
    hss_den = (hits + misses) * (misses + negatives) + (hits + false_alarms) * (
        false_alarms + negatives
    )

    def ratio(num: float, den: float) -> float:
        return num / den if den else float('nan')

    return {
        'wet_fraction': float(wet.mean()),
        'pod': ratio(hits, hits + misses),
        'far': ratio(false_alarms, hits + false_alarms),
        'csi': ratio(hits, hits + misses + false_alarms),
        'hss': ratio(2 * (hits * negatives - misses * false_alarms), hss_den),
        'total_ratio': ratio(float(values.sum()), float(obs.sum())),
    }


def skill(
    frame: pd.DataFrame,
    *,
    observed: str,
    precipitation: bool,
) -> dict[str, Any]:
    """Raw and corrected scores of ``frame[['raw', 'corrected']]`` against ``observed``."""
    frame = frame[np.isfinite(frame['corrected'])]
    obs = frame[observed].to_numpy(np.float64)
    result: dict[str, Any] = {
        'station_days': len(frame),
        'stations': int(frame['station_id'].nunique()),
        'observed_mean': float(obs.mean()) if len(obs) else float('nan'),
    }
    for kind in ('raw', 'corrected'):
        values = frame[kind].to_numpy(np.float64)
        scores = _continuous(obs, values)
        if precipitation:
            scores |= _wet_scores(obs, values)
        result[kind] = scores
    if precipitation:
        result['observed_wet_fraction'] = float((obs >= WET_THRESHOLD).mean())
    result['monthly'] = {
        int(month): {
            kind: _continuous(
                group[observed].to_numpy(np.float64), group[kind].to_numpy(np.float64)
            )
            for kind in ('raw', 'corrected')
        }
        | {'station_days': len(group)}
        for month, group in frame.groupby(frame['date'].dt.month)
    }
    return result


# ── Forecast consistency ────────────────────────────────────────────────────


def forecast_consistency(
    forecast: Source,
    root: zarr.Group,
    artifacts: FileStore,
    dem_store: Store,
    variable: str,
) -> dict[str, Any] | None:
    """Mean change the reanalysis models make to the forecast, next to the reanalysis.

    ``variable`` is ``'temp'`` or ``'precip'``. Returns ``None`` without a forecast.
    """
    if forecast.group_path not in root:
        return None
    model = forecast.model
    if variable == 'temp':
        var = model.temp_var
        corrector: Any = TemperatureCorrector(forecast, artifacts, dem_store).load_artifacts()
        scale = 1.0
    else:
        var = model.precip_var
        corrector = PrecipCorrector(forecast, artifacts, dem_store).load_artifacts()
        scale = model.precip_to_mm
    climate = corrector.climatology
    lats, lons = (climate.lats, climate.lons) if climate is not None else model.grid(root)

    def corrected_means(
        source: Source, indices: NDArray[np.intp]
    ) -> list[tuple[date, float, float, float, float]]:
        rows = []
        with open_regridded(source, root, lats, lons) as read:
            block = read((var,), indices)
            for index, value in enumerate(block.time.to_numpy().astype('datetime64[D]')):
                day = date.fromisoformat(str(value))
                raw = block[var].to_numpy()[index]
                corrected = corrector.correct_grid(day, raw, lats, lons)['corrected']
                raw = raw * scale if variable == 'precip' else raw
                valid = np.isfinite(raw) & np.isfinite(corrected)
                if not valid.any():
                    continue
                rows.append(
                    (
                        day,
                        float(raw[valid].mean()),
                        float(corrected[valid].mean()),
                        float((raw[valid] >= WET_THRESHOLD).mean()),
                        float((corrected[valid] >= WET_THRESHOLD).mean()),
                    )
                )
        return rows

    forecast_days = _days(forecast, root)
    if not len(forecast_days):
        return None
    sampled = np.arange(0, len(forecast_days), _SAMPLE_STEP)
    forecast_rows = corrected_means(forecast, sampled)
    months = sorted({day.month for day, *_ in forecast_rows})
    reanalysis_days = _days(model, root)
    reanalysis_rows = (
        corrected_means(model, _recent_month_days(reanalysis_days, months))
        if len(reanalysis_days)
        else []
    )
    return {
        'issued': str(root.attrs.get('updated', '')),
        'sampled_days': len(forecast_rows),
        'monthly': {
            month: {
                'forecast': _summarize(
                    [row for row in forecast_rows if row[0].month == month], variable
                ),
                'reanalysis': _summarize(
                    [row for row in reanalysis_rows if row[0].month == month], variable
                ),
            }
            for month in months
        },
    }


def _days(source: Source, root: zarr.Group) -> NDArray[np.datetime64]:
    if source.group_path not in root:
        return np.array([], 'datetime64[D]')
    with open_dataset(root.store, source.path(root)) as ds:
        return ds.time.to_numpy().astype('datetime64[D]')


def _recent_month_days(days: NDArray[np.datetime64], months: Sequence[int]) -> NDArray[np.intp]:
    """Pick a few days of each calendar month from the latest year that has it."""
    month_numbers = days.astype('datetime64[M]').astype(int) % 12 + 1
    years = days.astype('datetime64[Y]').astype(int)
    picked: list[NDArray[np.intp]] = []
    for month in months:
        candidates = np.flatnonzero(month_numbers == month)
        if not len(candidates):
            continue
        latest = candidates[years[candidates] == years[candidates].max()]
        step = max(len(latest) // _REANALYSIS_DAYS_PER_MONTH, 1)
        picked.append(latest[::step][:_REANALYSIS_DAYS_PER_MONTH])
    return np.sort(np.concatenate(picked)) if picked else np.array([], np.intp)


def _summarize(rows: list[tuple[date, float, float, float, float]], variable: str) -> dict:
    if not rows:
        return {'days': 0}
    values = np.asarray([row[1:] for row in rows], np.float64)
    summary = {
        'days': len(rows),
        'raw_mean': float(values[:, 0].mean()),
        'corrected_mean': float(values[:, 1].mean()),
        'shift': float((values[:, 1] - values[:, 0]).mean()),
    }
    if variable == 'precip':
        summary['raw_wet_fraction'] = float(values[:, 2].mean())
        summary['corrected_wet_fraction'] = float(values[:, 3].mean())
    return summary


# ── Report files ────────────────────────────────────────────────────────────


def update_report(artifacts: FileStore, source: str, section: str, data: dict[str, Any]) -> None:
    """Store one section of ``<source>/quality_report.json`` and refresh the Markdown."""
    directory = artifacts.child(source)
    directory.fs.makedirs(directory.root, exist_ok=True)
    report: dict[str, Any] = {}
    if directory.exists(REPORT_JSON):
        with directory.open(REPORT_JSON) as stream:
            report = json.load(stream)
    report['source'] = source
    report[section] = data | {'created': datetime.now(UTC).isoformat(timespec='seconds')}
    with directory.write_atomic(REPORT_JSON) as stream:
        stream.write(json.dumps(_clean(report), indent=2, sort_keys=True).encode('utf-8'))
    write_markdown(artifacts)


def write_markdown(artifacts: FileStore) -> None:
    """Combine the JSON reports of every trained source into one readable file."""
    reports = []
    for name in SOURCES:
        directory = artifacts.child(name)
        if directory.exists(REPORT_JSON):
            with directory.open(REPORT_JSON) as stream:
                reports.append(json.load(stream))
    lines = [
        '# Bias-correction quality',
        '',
        (
            'Scores compare daily values with station observations held out by '
            'station-grouped cross-validation: each station is scored by models trained '
            'without it. "Raw" is the reanalysis value at the station cell, "corrected" '
            'is the value after correction. Bias = model - observation.'
        ),
        '',
    ]
    for report in reports:
        title = _TITLES.get(report['source'], report['source'])
        for section, heading, unit, precipitation in (
            ('temp', 'temperature', 'degC', False),
            ('precip', 'precipitation', 'mm/day', True),
        ):
            if section in report:
                lines += _skill_markdown(
                    f'{title} reanalysis: {heading}',
                    report[section],
                    unit,
                    precipitation=precipitation,
                )
        for section, heading, unit in (
            ('forecast_temp', 'temperature', 'degC'),
            ('forecast_precip', 'precipitation', 'mm/day'),
        ):
            if section in report:
                lines += _forecast_markdown(
                    f'{title} forecast: {heading}', report[section], unit, section
                )
    with artifacts.write_atomic(REPORT_MARKDOWN) as stream:
        stream.write('\n'.join(lines).encode('utf-8'))


def _skill_markdown(title: str, data: dict, unit: str, *, precipitation: bool) -> list[str]:
    lines = [f'## {title}', '', f'Created {data["created"]}.', '']
    if 'skipped' in data:
        return [*lines, f'Not evaluated: {data["skipped"]}.', '']
    lines += [
        (
            f'{data["station_days"]} station-days at {data["stations"]} stations, '
            f'{data["folds"]}-fold cross-validation.'
        ),
        '',
        '| Metric | Raw | Corrected | Change |',
        '| --- | ---: | ---: | ---: |',
    ]
    rows: list[tuple[str, str, Callable[[float | None, float | None], str]]] = [
        (f'Bias, {unit}', 'bias', _abs_change),
        (f'MAE, {unit}', 'mae', _relative_change),
        (f'RMSE, {unit}', 'rmse', _relative_change),
        ('Correlation', 'corr', _difference),
    ]
    if precipitation:
        rows += [
            ('Wet-day fraction', 'wet_fraction', _difference),
            ('POD', 'pod', _difference),
            ('FAR', 'far', _difference),
            ('CSI', 'csi', _difference),
            ('HSS', 'hss', _difference),
            ('Total / observed total', 'total_ratio', _difference),
        ]
    for label, key, change in rows:
        raw, corrected = data['raw'][key], data['corrected'][key]
        lines.append(f'| {label} | {_fmt(raw)} | {_fmt(corrected)} | {change(raw, corrected)} |')
    if precipitation:
        lines += ['', f'Observed wet-day fraction: {_fmt(data["observed_wet_fraction"])}.']
    lines += [
        '',
        f'By month, {unit}:',
        '',
        '| Month | Days | Bias raw | Bias corr. | MAE raw | MAE corr. | RMSE raw | RMSE corr. |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for month, scores in sorted(data['monthly'].items(), key=lambda item: int(item[0])):
        raw, corrected = scores['raw'], scores['corrected']
        lines.append(
            f'| {month} | {scores["station_days"]} | {_fmt(raw["bias"])} | '
            f'{_fmt(corrected["bias"])} | {_fmt(raw["mae"])} | {_fmt(corrected["mae"])} | '
            f'{_fmt(raw["rmse"])} | {_fmt(corrected["rmse"])} |'
        )
    return [*lines, '']


def _forecast_markdown(title: str, data: dict, unit: str, section: str) -> list[str]:
    lines = [
        f'## {title}',
        '',
        (
            f'Created {data["created"]} from the forecast issued {data["issued"]} '
            f'({data["sampled_days"]} sampled days). Forecast errors are not evaluated: '
            'the table shows the mean change of the domain-mean value made by the reanalysis '
            'models, and the change they make to the reanalysis in the same calendar month '
            '(latest available year). Similar shifts mean the systematic reanalysis bias is '
            'removed from the forecast consistently.'
        ),
        '',
    ]
    precipitation = section == 'forecast_precip'
    header = f'| Month | Forecast raw, {unit} | Forecast corr., {unit} | Forecast shift | '
    header += 'Reanalysis shift |'
    divider = '| ---: | ---: | ---: | ---: | ---: |'
    if precipitation:
        header += ' Wet fraction raw | Wet fraction corr. |'
        divider += ' ---: | ---: |'
    lines += [header, divider]
    for month, entry in sorted(data['monthly'].items(), key=lambda item: int(item[0])):
        fc, rean = entry['forecast'], entry['reanalysis']
        row = (
            f'| {month} | {_fmt(fc.get("raw_mean"))} | {_fmt(fc.get("corrected_mean"))} | '
            f'{_fmt(fc.get("shift"))} | {_fmt(rean.get("shift"))} |'
        )
        if precipitation:
            row += (
                f' {_fmt(fc.get("raw_wet_fraction"))} | {_fmt(fc.get("corrected_wet_fraction"))} |'
            )
        lines.append(row)
    return [*lines, '']


def _finite(*values: float | None) -> bool:
    """Check scores read back from JSON, where NaN is stored as null."""
    return all(value is not None and np.isfinite(value) for value in values)


def _fmt(value: float | None) -> str:
    return f'{value:.3f}' if value is not None and _finite(value) else '—'


def _relative_change(raw: float | None, corrected: float | None) -> str:
    if raw is None or corrected is None or not _finite(raw, corrected) or raw == 0:
        return '—'
    return f'{(corrected - raw) / raw * 100:+.1f}%'


def _abs_change(raw: float | None, corrected: float | None) -> str:
    if raw is None or corrected is None or not _finite(raw, corrected):
        return '—'
    return f'|bias| {abs(corrected) - abs(raw):+.3f}'


def _difference(raw: float | None, corrected: float | None) -> str:
    if raw is None or corrected is None or not _finite(raw, corrected):
        return '—'
    return f'{corrected - raw:+.3f}'


def _clean(value: Any) -> Any:
    """Plain JSON: string keys, Python scalars and null instead of NaN."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
