from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from httpx2 import Client
from zarr.abc.store import Store

from weatheasy import const, download
from weatheasy.query import Coords, get_cfs2_data, get_cmip6_data
from weatheasy.raw_cache import DownloadCache
from weatheasy.storage import get_store
from weatheasy.util import get_storage


@pytest.mark.parametrize('hour', [0, 12])
def test_cmip6_download_getter_resume_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hour: int,
):
    monkeypatch.setattr(const, 'CMIP6_FIRST_YEAR', 2000)
    monkeypatch.setattr(const, 'CMIP6_LAST_YEAR', 2000)
    monkeypatch.setattr(const, 'CMIP6_VARS', {'tas': const.CMIP6_VARS['tas']})
    times = np.arange(np.datetime64('2000-01-01'), np.datetime64('2001-01-01')) + np.timedelta64(
        hour, 'h'
    )
    values = np.full((366, 3, 2), 273.15, dtype='float32')
    values[:, 2] = 283.15
    values[59, 2] = 290
    source = tmp_path / 'source.nc'
    xr.Dataset(
        {'tas': (('time', 'lat', 'lon'), values, {'units': 'K'})},
        coords={'time': times, 'lat': [-25.125, 0.125, 55.125], 'lon': [0.125, 0.375]},
    ).to_netcdf(source)
    calls = []

    def load(*args):
        calls.append(args[2])
        return source

    monkeypatch.setattr(download, '_load_cmip6_dataset', load)
    root = get_storage(str(tmp_path / 'data'))
    download.download_cmip6_data(get_store(tmp_path / 'data'))
    download.download_cmip6_data(root.store)
    assert calls == [2000]
    result = get_cmip6_data(
        store=root.store,
        begin=date(2000, 2, 28),
        end=date(2000, 3, 1),
        coords=Coords(55.125, 0.125),
        variables=['tas'],
    )
    np.testing.assert_allclose(result, [[283.15, 290, 283.15]])
    np.testing.assert_allclose(
        get_cmip6_data(
            store=get_store(tmp_path / 'data'),
            begin=date(2000, 2, 28),
            end=date(2000, 3, 1),
            coords=Coords(55.125, 0.125),
            variables=['tas'],
        ),
        [[283.15, 290, 283.15]],
    )
    from weatheasy.__main__ import main

    target = tmp_path / 'result.csv'
    monkeypatch.setattr(
        'sys.argv',
        [
            'weatheasy',
            'query',
            'cmip6',
            '-d',
            str(tmp_path / 'data'),
            '-o',
            str(target),
            '2000-02-29',
            '2000-02-29',
            '55.125',
            '0.125',
            'tas',
        ],
    )
    main(configure_logging=False)
    assert target.read_text() == 'DATE,tas\n2000-02-29,290\n'


def write_grib(path: Path, when: datetime, values: list[tuple[int, ...]]):
    ec = pytest.importorskip('eccodes')
    with path.open('wb') as stream:
        for item in values:
            category, number, value = (0, *item) if len(item) == 2 else item
            h = ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
            for key, val in {
                'Ni': 3,
                'Nj': 2,
                'latitudeOfFirstGridPointInDegrees': 56,
                'latitudeOfLastGridPointInDegrees': 55,
                'longitudeOfFirstGridPointInDegrees': 0,
                'longitudeOfLastGridPointInDegrees': 2,
                'iDirectionIncrementInDegrees': 1,
                'jDirectionIncrementInDegrees': 1,
                'shapeOfTheEarth': 6,
                'dataDate': int(when.strftime('%Y%m%d')),
                'dataTime': when.hour * 100,
                'discipline': 0,
                'parameterCategory': category,
                'parameterNumber': number,
                'typeOfFirstFixedSurface': 103,
                'scaledValueOfFirstFixedSurface': 2,
            }.items():
                ec.codes_set(h, key, val)
            ec.codes_set_values(h, np.full(6, value))
            ec.codes_write(h, stream)
            ec.codes_release(h)


def forecast(
    begin: date,
    end: date,
    _session: Client,
    download_dir: DownloadCache,
):
    day = begin
    while day < end:
        for kind in ('flx', 'pgb'):
            for hour in (0, 6, 12, 18):
                path = download_dir.file(f'{kind}{day:%Y%m%d}{hour:02d}.grb2')
                write_grib(
                    path,
                    datetime.combine(day, datetime.min.time()).replace(hour=hour),
                    [(0, 290), (4, 310 + hour), (5, 270 - hour), (1, 1, 60)],
                )
        day += const.ONE_DAY


@pytest.mark.parametrize('missing_reanalysis', [False, True])
def test_cfs2_public_download_extrema_missing_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_reanalysis: bool,
):
    # Min and max are intentionally different fields; sharing the old band fails.
    flx = {name: const.CFS2_FLX_BANDS[name] for name in ('TMP', 'TMIN', 'TMAX')}
    monkeypatch.setattr(const, 'CFS2_FLX_BANDS', flx)
    monkeypatch.setattr(const, 'CFS2_REANALYSIS_FIRST_DATE', date(2026, 9, 18))
    monkeypatch.setattr(const, 'CFS2_FORECAST_DAYS', const.ONE_DAY * 2)
    monkeypatch.setattr(download, 'utc_now', lambda: datetime(2026, 9, 22, tzinfo=UTC))
    calls = []

    def reanalysis(_self: download._Cfs2ReanalysisDownloader, day: date):
        calls.append(day)
        if missing_reanalysis:
            return None
        paths = []
        for hour in (0, 6, 12, 18):
            path = tmp_path / f'r{day}-{hour}.grb2'
            write_grib(
                path,
                datetime.combine(day, datetime.min.time()).replace(hour=hour),
                [(0, 280), (4, 300 + hour), (5, 260 - hour), (1, 1, 50)],
            )
            paths.append(path)
        return paths

    monkeypatch.setattr(
        download._Cfs2ReanalysisDownloader,
        '__call__',
        reanalysis,
    )

    monkeypatch.setattr(download, '_download_cfs2_forecast', forecast)
    root = get_storage(str(tmp_path / 'data'))
    download.download_cfs2_data(get_store(tmp_path / 'data'))
    download.download_cfs2_data(root.store)
    assert calls == [date(2026, 9, 18)]
    result = get_cfs2_data(
        store=root.store,
        begin=date(2026, 9, 18),
        end=date(2026, 9, 19),
        coords=Coords(55, 0),
        variables=['TMP', 'TMIN', 'TMAX', 'RH'],
    )
    expected = np.array([[6.85, 16.85], [-31.15, -21.15], [44.85, 54.85], [50, 60]])
    if missing_reanalysis:
        expected[:, 0] = np.nan
        from weatheasy.storage import StorageError

        target = root['reanalysis/flx']
        target.attrs['write_state'] = 'writing'
        with pytest.raises(StorageError, match='completed'):
            get_cfs2_data(
                store=root.store,
                begin=date(2026, 9, 18),
                end=date(2026, 9, 18),
                coords=Coords(55, 0),
                variables=['TMP'],
            )
        with pytest.raises(StorageError, match='completed'):
            get_cfs2_data(
                store=get_store(tmp_path / 'data'),
                begin=date(2026, 9, 18),
                end=date(2026, 9, 18),
                coords=Coords(55, 0),
                variables=['TMP'],
            )
        target.attrs.clear()
    np.testing.assert_allclose(result, expected, atol=1e-4)
    np.testing.assert_allclose(
        get_cfs2_data(
            store=get_store(tmp_path / 'data'),
            begin=date(2026, 9, 18),
            end=date(2026, 9, 19),
            coords=Coords(55, 0),
            variables=['TMP', 'TMIN', 'TMAX', 'RH'],
        ),
        expected,
        atol=1e-4,
    )
    check_forecast_replacement(root, monkeypatch, missing_reanalysis)


def check_forecast_replacement(
    root: zarr.Group,
    monkeypatch: pytest.MonkeyPatch,
    missing_reanalysis: bool,  # noqa: FBT001
):
    # A sentinel verifies deletion of the previous forecast before any new write.
    root['forecast'].attrs['old_forecast'] = True
    previous_reanalysis_attrs = dict(root['reanalysis/flx'].attrs)
    original_write = download.write_batch
    monkeypatch.setattr(download, 'utc_now', lambda: datetime(2026, 9, 23, tzinfo=UTC))

    def fail_second_grid(
        store: Store,
        group: str,
        dataset: xr.Dataset,
    ):
        if group.startswith('forecast/'):
            assert const.CFS2_KEY_UPDATED not in root[''].attrs
            assert 'old_forecast' not in root['forecast'].attrs
        if group == 'forecast/pgb':
            msg = 'simulated interrupted second-grid write'
            raise OSError(msg)
        return original_write(store, group, dataset)

    with monkeypatch.context() as patch:
        patch.setattr(download, 'write_batch', fail_second_grid)
        with pytest.raises(OSError, match='second-grid'):
            download.download_cfs2_data(root.store)
        assert const.CFS2_KEY_UPDATED not in root[''].attrs
        from weatheasy.error import CFS2Error

        with pytest.raises(CFS2Error):
            get_cfs2_data(
                store=root.store,
                begin=date(2026, 9, 23),
                end=date(2026, 9, 23),
                coords=Coords(55, 0),
                variables=['TMP'],
            )
        # Reanalysis data is appended independently, never deleted with forecast.
        assert 'reanalysis/flx' in root
        if not missing_reanalysis:
            assert (
                root['reanalysis/flx'].attrs['committed_length']
                >= previous_reanalysis_attrs['committed_length']  # ty: ignore[unsupported-operator]
            )
    download.download_cfs2_data(root.store)
    assert root[''].attrs['updated'] == '2026-09-23'
    assert set(root.get_group('forecast').group_keys()) == {'flx', 'pgb'}
    # Direct consumers use a permanent group path; no pointer lookup is needed.
    with xr.open_zarr(root.store, group='forecast/flx', consolidated=False, chunks=None) as direct:
        np.testing.assert_allclose(direct.TMP.to_numpy(), 16.85, atol=1e-4)
    actual = get_cfs2_data(
        store=root.store,
        begin=date(2026, 9, 23),
        end=date(2026, 9, 23),
        coords=Coords(55, 0),
        variables=['TMP', 'RH'],
    )
    np.testing.assert_allclose(actual, [[16.85], [60]], atol=1e-4)


def test_http_prepared_data_uses_getter_without_prepare_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sys

    pytest.importorskip('fastapi')
    from starlette.testclient import TestClient

    from weatheasy.storage import prepare_dataset, write_batch
    from weatheasy.web import create_app
    from weatheasy.web.config import Settings, SourceSettings, SourcesSettings

    root = get_storage(str(tmp_path / 'store'))
    ds = xr.Dataset(
        {
            'tas': (
                ('time', 'latitude', 'longitude'),
                np.full((1, 2, 2), 280, dtype='float32'),
                {'units': 'K'},
            )
        },
        coords={
            'time': [np.datetime64('2000-01-01')],
            'latitude': [54.0, 55.0],
            'longitude': [0.0, 1.0],
        },
    )
    write_batch(root.store, 'tas', prepare_dataset(ds, crs='EPSG:4326'))
    app = create_app(
        Settings(
            sources=SourcesSettings(cmip6=SourceSettings(store=str(tmp_path / 'store'))),
            _env_file=None,
        )
    )
    monkeypatch.setitem(sys.modules, 'cfgrib', None)
    monkeypatch.setitem(sys.modules, 'eccodes', None)

    with TestClient(app) as client:
        response = client.get(
            '/cmip6',
            params={
                'begin': '2000-01-01',
                'end': '2000-01-01',
                'lat': 55,
                'lon': 0,
                'var': 'tas',
            },
        )
    assert response.status_code == 200, response.text
    assert response.json() == [{'tas': 280, 'date': '2000-01-01'}]


def test_daily_missing_forecast_sample_remains_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(const, 'CFS2_FLX_BANDS', {'TMP': const.CFS2_FLX_BANDS['TMP']})
    path = tmp_path / 'sample.grb2'
    write_grib(path, datetime(2026, 9, 22, tzinfo=UTC), [(0, 280)])
    daily = download._daily_cfs2([path], 'flx', date(2026, 9, 22))
    assert np.isnan(daily.TMP.to_numpy()).all()
