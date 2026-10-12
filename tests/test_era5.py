import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from weatheasy import const, era5
from weatheasy.era5_grib import read_grib
from weatheasy.query import Coords, get_era5_data
from weatheasy.storage import StorageError, get_store, open_dataset
from weatheasy.util import get_storage


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local-app-data'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('USERPROFILE', str(tmp_path / 'home'))


FIXTURES = Path(__file__).parent / 'fixtures' / 'era5'


def source(month: date, *, boundary: bool = False):
    first = np.datetime64(month, 'h')
    times = (
        [first]
        if boundary
        else np.arange(first, np.datetime64(era5.next_month(month), 'h'), np.timedelta64(1, 'h'))
    )
    values = {}
    defaults = {
        't2m': 293.15,
        'd2m': 293.15,
        'sp': 100000,
        'u10': 3,
        'v10': 4,
        'tp': 0.008,
        'pev': -0.004,
        'slhf': -8640000,
        'sshf': -4320000,
        'ssrd': 8640000,
        'ssr': 6480000,
        'strd': 17280000,
        'str': -864000,
        'sde': 0.1,
        'sd': 0.02,
    }
    names = {v.nc_name for v in const.ERA5_VARS.values()} | set(era5.EXTRA)
    for name in names:
        val = defaults.get(name, 280 if name.startswith(('stl', 'skt')) else 0.2)
        array = np.full((len(times), 2, 3), val, dtype='float64')
        if name in ('u10', 'v10') and not boundary:
            array[1::2] *= -1
        values[name] = (('time', 'latitude', 'longitude'), array)
    return xr.Dataset(
        values, coords={'time': times, 'latitude': [55.3, 55.2], 'longitude': [37.1, 37.2, 37.3]}
    ).rio.write_crs('+proj=longlat +R=6367470 +no_defs')


class FakeCDSClient:
    __slots__ = ('calls', 'jobs', 'resumed', 'transform')

    def __init__(self) -> None:
        self.calls = []
        self.jobs = {}
        self.resumed = []
        self.transform: Callable[[xr.Dataset, Mapping, int], xr.Dataset] = (
            lambda ds, _request, _call: ds
        )

    def submit(self, _dataset: str, request: Mapping):
        assert request['data_format'] == 'grib'
        assert len(request['variable']) <= 8
        assert len(request['variable']) * len(request['day']) * len(request['time']) * 2 <= 12000
        self.calls.append(request)
        call = len(self.calls)
        # Transport tests use a lightweight response; static GRIB tests below exercise cfgrib.
        remote = SimpleNamespace(
            request_id=f'job-{call}',
            status='successful',
            results_ready=True,
            download=lambda path: self._download(request, call, path),
        )
        self.jobs[remote.request_id] = remote
        return remote

    def get_remote(self, request_id: str):
        self.resumed.append(request_id)
        return self.jobs[request_id]

    def _download(self, request: Mapping, call: int, path: str) -> None:
        month = date(int(request['year']), int(request['month']), 1)
        ds = self.transform(requested_source(month, request), request, call)
        ds.to_netcdf(path, engine='netcdf4')


@pytest.fixture
def cds_client(monkeypatch: pytest.MonkeyPatch):
    def read_response(path, required):  # noqa: ANN001
        with xr.open_dataset(path, engine='netcdf4', decode_coords='all') as ds:
            return ds[sorted(required)].load()

    monkeypatch.setattr(era5, 'read_grib', read_response)
    client = FakeCDSClient()
    monkeypatch.setitem(
        sys.modules,
        'cdsapi',
        SimpleNamespace(Client=lambda **_kwargs: SimpleNamespace(client=client)),
    )
    return client


@pytest.mark.parametrize('month', [date(2000, 2, 1), date(2000, 12, 1)])
def test_daily_units_midnight_wind_and_coordinates(month: date):
    main = source(month)
    boundary = source(era5.next_month(month), boundary=True)
    # 23 UTC is deliberately wrong; only the NEXT midnight carries full accumulation.
    main['tp'][23::24] = 0.001
    boundary['tp'][:] = 0.012
    main['t2m'][0, 0, 0] = np.nan
    result = xr.concat(
        [
            era5.process_day(main, boundary, month + i * const.ONE_DAY)
            for i in range((era5.next_month(month) - month).days)
        ],
        dim='time',
    )
    assert result.sizes['time'] == (era5.next_month(month) - month).days
    np.testing.assert_allclose(result.WIND, 5)
    np.testing.assert_allclose(result.UGRD, 0)
    np.testing.assert_allclose(result.VGRD, 0)
    np.testing.assert_allclose(result.PRATE[:-1] * 86400, 8)
    np.testing.assert_allclose(result.PRATE[-1] * 86400, 12)
    for var, expected in {
        'PRES': 1000,
        'PEVPR': 4,
        'LHTFL': 100,
        'SHTFL': 50,
        'DSWRF': 100,
        'DLWRF': 200,
        'USWRF': 25,
        'ULWRF': 210,
        'WEASD': 20,
    }.items():
        np.testing.assert_allclose(result[var], expected)
    np.testing.assert_allclose(result.RH[1:], 100)
    assert np.isnan(result.TMP[0, 1, 0])
    assert result.spatial_ref.attrs['semi_major_axis'] == 6371229
    np.testing.assert_allclose(main.skt, 280)  # Reduction must not modify its input arrays.
    np.testing.assert_allclose(result.latitude, [55.2, 55.3])


# Required ERA5-Land fields, independently of the production variable registry.
REQUIRED_INPUTS = (
    'd2m',
    'pev',
    'sd',
    'sde',
    'skt',
    'slhf',
    'sp',
    'sshf',
    'ssr',
    'ssrd',
    'stl1',
    'stl2',
    'stl3',
    'stl4',
    'str',
    'strd',
    'swvl1',
    'swvl2',
    'swvl3',
    'swvl4',
    't2m',
    'tp',
    'u10',
    'v10',
)


def test_boundary_grid_validation():
    main = source(date(2000, 12, 1))
    boundary = source(date(2001, 1, 1), boundary=True)
    with pytest.raises(StorageError, match='coordinates'):
        era5._validate_boundary(main, boundary.assign_coords(longitude=[38.1, 38.2, 38.3]))


def test_public_monthly_download_resume_getter_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cds_client: FakeCDSClient,
):
    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    root = get_storage(str(tmp_path / 'store'))
    bbox = (55.3, 37.1, 55.2, 37.3)
    era5.download_era5_data(get_store(tmp_path / 'store'), tmp_path / 'cache', bbox=bbox)
    assert len(cds_client.calls) == 6
    assert all(request['area'] == list(bbox) for request in cds_client.calls)
    assert root[''].attrs['request_bbox'] == list(bbox)
    assert root[''].attrs['last_month'] == [2000, 1]
    assert root[''].attrs['era5_requests'] == {}
    root[''].attrs['last_month'] = [1999, 12]  # crash before convenience marker
    root[''].attrs['era5_requests'] = {'committed-month': 'job-6'}
    era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert len(cds_client.calls) == 6
    assert root[''].attrs['last_month'] == [2000, 1]
    assert root[''].attrs['era5_requests'] == {}
    values = get_era5_data(
        store=root.store,
        begin=date(2000, 1, 31),
        end=date(2000, 1, 31),
        coords=Coords(55.3, 37.2),
        variables=['WIND', 'TMP'],
    )
    np.testing.assert_allclose(values, [[5], [20]], atol=1e-5)
    before = dict(root[''].attrs)
    with pytest.raises(StorageError, match='different requested bbox'):
        era5.download_era5_data(get_store(tmp_path / 'store'), bbox=(55.3, 38.1, 55.2, 38.3))
    assert dict(root[''].attrs) == before
    with open_dataset(root.store, 'daily') as ds:
        assert ds.attrs['request_bbox'] == list(bbox)
        assert ds.spatial_ref.attrs['semi_major_axis'] == 6371229
        assert ds.WIND.attrs['units'] == 'm/s'
    from weatheasy.__main__ import main

    output = tmp_path / 'result.csv'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'weatheasy',
            'query',
            'era5',
            '-d',
            str(tmp_path / 'store'),
            '-o',
            str(output),
            '2000-01-31',
            '2000-01-31',
            '55.3',
            '37.2',
            'WIND',
        ],
    )
    main(configure_logging=False)
    assert output.read_text() == 'DATE,WIND\n2000-01-31,5\n'


def test_failed_month_and_append_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cds_client: FakeCDSClient
):
    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 3, 10, tzinfo=UTC))
    broken = True

    def corrupt_march(ds: xr.Dataset, request: Mapping, _call: int) -> xr.Dataset:
        return ds.drop_vars(list(ds.data_vars)) if broken and request['month'] == '03' else ds

    cds_client.transform = corrupt_march
    root = get_storage(str(tmp_path / 'store'))
    bbox = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(KeyError):
        era5.download_era5_data(root.store, bbox=bbox)
    assert root[''].attrs['last_month'] == [2000, 1]
    assert era5.available_period(root.store)[1] == date(2000, 1, 31)
    assert len(root[''].attrs['era5_requests']) == 6
    broken = False
    original = xr.Dataset.to_zarr

    def fail_write(self, *args, **kwargs):  # noqa: ANN001
        original(self, *args, **kwargs)
        msg = 'interrupted after appending arrays'
        raise OSError(msg)

    monkeypatch.setattr(xr.Dataset, 'to_zarr', fail_write)
    with pytest.raises(OSError, match='interrupted'):
        era5.download_era5_data(root.store, bbox=bbox)
    assert root[''].attrs['last_month'] == [2000, 1]
    monkeypatch.setattr(xr.Dataset, 'to_zarr', original)
    era5.download_era5_data(root.store, bbox=bbox)
    assert era5.available_period(root.store) == (date(2000, 1, 1), date(2000, 2, 29))
    assert root['daily'].attrs['committed_length'] == 60
    assert len(cds_client.calls) == 12  # Six jobs per month, including retries after crashes.
    assert root[''].attrs['era5_requests'] == {}


def test_cache_includes_bbox(tmp_path: Path, cds_client: FakeCDSClient):
    client = cds_client
    for bbox in [(55.3, 37.1, 55.2, 37.3), (55.3, 38.1, 55.2, 38.3)]:
        era5._Downloader(client, tmp_path, bbox, get_store(tmp_path / 'store')).retrieve(
            date(2000, 1, 1), boundary=False
        )
    assert len(client.calls) == 6
    assert len(list(tmp_path.glob('*.grib'))) == 6


@pytest.mark.parametrize(
    ('state', 'replacement'),
    [
        ('accepted', 0),
        ('running', 0),
        ('successful', 0),
        ('failed', 1),
        (404, 1),
        (410, 1),
        (401, None),
        (503, None),
    ],
)
def test_restart_resumes_persisted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cds_client: FakeCDSClient,
    state: str | int,
    replacement: int | None,
):
    from requests import HTTPError, Response

    store_path = tmp_path / 'store'
    bbox = (55.3, 37.1, 55.2, 37.3)
    download = FakeCDSClient._download

    def interrupt(_self, _request, _call, path):  # noqa: ANN001
        # The journal must already be on disk before the SDK waits/downloads.
        saved = era5.group_attrs(get_store(store_path), '')['era5_requests']
        assert list(saved.values()) == ['job-1']
        Path(path).write_bytes(b'partial download')
        msg = 'download interrupted'
        raise OSError(msg)

    monkeypatch.setattr(FakeCDSClient, '_download', interrupt)
    with pytest.raises(OSError, match='download interrupted'):
        era5._Downloader(cds_client, tmp_path / 'cache', bbox, get_store(store_path), 1).retrieve(
            date(2000, 1, 1), boundary=True
        )
    saved = era5.group_attrs(get_store(store_path), '')['era5_requests']
    monkeypatch.setattr(FakeCDSClient, '_download', download)
    if isinstance(state, int):

        def lookup_error(_self, request_id):  # noqa: ANN001
            assert request_id == 'job-1'
            response = Response()
            response.status_code = state
            raise HTTPError(response=response)

        monkeypatch.setattr(FakeCDSClient, 'get_remote', lookup_error)
    else:
        cds_client.jobs['job-1'].status = state
    # A new downloader/store handle must reconstruct its state from Zarr.
    restarted = era5._Downloader(cds_client, tmp_path / 'cache', bbox, get_store(store_path), 1)
    if replacement is None:
        with pytest.raises(HTTPError):
            restarted.retrieve(date(2000, 1, 1), boundary=True)
        assert len(cds_client.calls) == 1
        assert era5.group_attrs(get_store(store_path), '')['era5_requests'] == saved
    else:
        result = restarted.retrieve(date(2000, 1, 1), boundary=True)
        assert len(cds_client.calls) == 3 + replacement
        assert result.sizes['time'] == 1
        current = era5.group_attrs(get_store(store_path), '')['era5_requests']
        assert len(current) == 3
        assert ('job-1' not in current.values()) == bool(replacement)
        assert len(list((tmp_path / 'cache').glob('*.grib'))) == 3


def test_grib_single_midnight():
    ds = read_grib(
        FIXTURES / 'midnight.grib',
        set(REQUIRED_INPUTS),
    )
    assert ds.sizes['time'] == 1
    np.testing.assert_allclose(ds.tp, 0.012, atol=1e-7)
    assert ds.rio.crs.to_dict()['R'] == 6371229


@pytest.mark.parametrize('different_coordinates', [False, True])
def test_grib_uses_product_crs_and_actual_grid(tmp_path: Path, *, different_coordinates: bool):
    import eccodes

    path = tmp_path / 'mixed.grib'
    editions = set()
    with (FIXTURES / 'midnight.grib').open('rb') as source, path.open('wb') as target:
        while (message := eccodes.codes_grib_new_from_file(source)) is not None:
            try:
                if eccodes.codes_get(message, 'cfVarName') == 't2m':
                    eccodes.codes_set(message, 'edition', 1)
                    if different_coordinates:
                        eccodes.codes_set(message, 'longitudeOfFirstGridPointInDegrees', 38.1)
                        eccodes.codes_set(message, 'longitudeOfLastGridPointInDegrees', 38.3)
                if eccodes.codes_get(message, 'cfVarName') == 'sde':
                    eccodes.codes_set(message, 'shapeOfTheEarth', 8)
                editions.add(eccodes.codes_get(message, 'edition'))
                eccodes.codes_write(message, target)
            finally:
                eccodes.codes_release(message)
    assert editions == {1, 2}
    if different_coordinates:
        with pytest.raises(ValueError, match='longitude'):
            read_grib(path, set(REQUIRED_INPUTS))
    else:
        ds = read_grib(path, set(REQUIRED_INPUTS))
        np.testing.assert_allclose(ds.t2m, 293.15, atol=1e-4)
        np.testing.assert_allclose(ds.tp, 0.012, atol=1e-7)
        np.testing.assert_allclose(ds.latitude, [55.3, 55.2])
        np.testing.assert_allclose(ds.longitude, [37.1, 37.2, 37.3])
        assert ds.rio.crs.to_dict()['R'] == 6371229
        assert ds.t2m.attrs['GRIB_edition'] == 1
        assert ds.t2m.attrs['GRIB_radius'] == 6367470
        assert ds.tp.attrs['GRIB_radius'] == 6367470
        assert ds.sde.attrs['GRIB_edition'] == 2
        assert ds.sde.attrs['GRIB_radius'] == 6371200
        np.testing.assert_allclose(ds.sde, 0.1, atol=1e-7)


def test_grib_ignores_unrequested_variables():
    ds = read_grib(FIXTURES / 'midnight.grib', {'t2m'})
    assert set(ds.data_vars) == {'t2m'}
    np.testing.assert_allclose(ds.t2m, 293.15, atol=1e-4)


def test_grib_uses_source_timestamps():
    ds = read_grib(FIXTURES / 'day.grib', {'t2m'})
    np.testing.assert_array_equal(
        ds.time,
        np.arange('2000-01-01T00', '2000-01-02T01', dtype='datetime64[h]'),
    )


def test_grib_missing_hour_preserves_other_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import eccodes
    from cfgrib import dataset as grib_dataset

    decoded = []
    decode = grib_dataset.get_values_in_order

    def decode_field(message, shape):  # noqa: ANN001
        decoded.append(message['cfVarName'])
        return decode(message, shape)

    monkeypatch.setattr(grib_dataset, 'get_values_in_order', decode_field)

    path = tmp_path / 'missing-hour.grib'
    with (FIXTURES / 'day.grib').open('rb') as source, path.open('wb') as target:
        while (message := eccodes.codes_grib_new_from_file(source)) is not None:
            try:
                if (
                    eccodes.codes_get(message, 'cfVarName') == 't2m'
                    and eccodes.codes_get(message, 'validityTime') == 0
                    and eccodes.codes_get(message, 'validityDate') == 20000101
                ):
                    continue
                eccodes.codes_write(message, target)
            finally:
                eccodes.codes_release(message)
    main = read_grib(path, set(REQUIRED_INPUTS))
    boundary = read_grib(FIXTURES / 'midnight.grib', set(REQUIRED_INPUTS))
    assert not decoded
    assert main.sizes['time'] == 25
    assert np.isnan(main.t2m.isel(time=0)).all()
    assert not decoded
    np.testing.assert_allclose(main.t2m.isel(time=1), 293.15, atol=1e-4)
    assert decoded == ['t2m']
    selected = main.t2m.isel(time=[0, 2], latitude=0, longitude=[2, 0]).to_numpy()
    assert np.isnan(selected[0]).all()
    np.testing.assert_allclose(selected[1], 293.15, atol=1e-4)
    np.testing.assert_allclose(main.skt.isel(time=0), 280)
    daily = era5.process_day(main, boundary, date(2000, 1, 1))
    for name in ('TMP', 'TMIN', 'TMAX', 'RH'):
        assert np.isnan(daily[name]).all()
    np.testing.assert_allclose(daily.SPFH, 0.01467417, atol=1e-8)
    np.testing.assert_allclose(daily.WIND, 5)
    np.testing.assert_allclose(daily.PRATE * 86400, 12)
    main.close()
    boundary.close()


@pytest.mark.parametrize('whole_hour', [False, True])
def test_monthly_download_keeps_missing_hour_as_nan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cds_client: FakeCDSClient,
    *,
    whole_hour: bool,
):
    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))

    def missing_hour(ds: xr.Dataset, request: Mapping, call: int) -> xr.Dataset:
        if request['month'] == '01' and (whole_hour or call == 1):
            return ds.isel(time=slice(1, None))
        return ds

    cds_client.transform = missing_hour
    store = get_store(tmp_path / 'store')
    era5.download_era5_data(store, tmp_path / 'cache', bbox=(55.3, 37.1, 55.2, 37.3))
    with open_dataset(store, 'daily') as daily:
        assert daily.sizes['time'] == 31
        assert np.isnan(daily.TMP.isel(time=0)).all()
        np.testing.assert_allclose(daily.TMP.isel(time=slice(1, None)), 20)
        np.testing.assert_allclose(daily.PEVPR, 4)
        if whole_hour:
            assert np.isnan(daily.PRES.isel(time=0)).all()
        else:
            np.testing.assert_allclose(daily.PRES, 1000)


def test_download_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys

    from weatheasy._cli import main

    calls = []
    monkeypatch.setattr(
        era5, 'download_era5_data', lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'weatheasy',
            'download',
            'era5',
            '-d',
            str(tmp_path / 'store'),
            '--bbox',
            '55.3',
            '37.1',
            '55.2',
            '37.3',
        ],
    )
    main(configure_logging=False)
    assert calls[0][1]['bbox'] == (55.3, 37.1, 55.2, 37.3)
    assert calls[0][1]['max_pending_requests'] == 3


def requested_source(month: date, request: Mapping):
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
    names.update({cds: nc for nc, cds in era5.EXTRA.items()})
    return source(month, boundary=len(request['time']) == 1)[
        [names[name] for name in request['variable']]
    ]


def test_partial_request_missing_variable_preserves_raw_but_never_publishes_daily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cds_client: FakeCDSClient,
):
    import zarr

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    broken = True

    def drop_variable(ds: xr.Dataset, _request: Mapping, call: int) -> xr.Dataset:
        return ds.drop_vars([next(iter(ds.data_vars))]) if broken and call == 2 else ds

    cds_client.transform = drop_variable
    read = era5.read_grib
    read_paths = []

    def track_read(path, required):  # noqa: ANN001
        read_paths.append(path)
        return read(path, required)

    monkeypatch.setattr(era5, 'read_grib', track_read)
    root = get_storage(str(tmp_path / 'store'))
    bbox = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(KeyError):
        era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert 'last_month' not in root[''].attrs
    assert 'daily' not in root
    assert len(list((tmp_path / 'cache/era5').glob('*.grib'))) == 2
    # Failed reads retain raw responses; retry encounters the same missing variable.
    with pytest.raises(KeyError):
        era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert len(cds_client.calls) == 5
    assert 'daily' not in root
    # Explicitly discard the invalid response before fetching corrected source data.
    read_paths[-1].unlink()
    broken = False
    era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert len(cds_client.calls) == 6  # Reuse the interrupted job as well as the cached file.
    assert 'job-2' in cds_client.resumed
    readonly = zarr.open_group(str(tmp_path / 'store'), mode='r')
    result = get_era5_data(
        store=readonly.store,
        begin=date(2000, 1, 1),
        end=date(2000, 1, 1),
        coords=Coords(55.3, 37.2),
        variables=['WIND'],
    )
    np.testing.assert_allclose(result, [[5]])


def test_cache_publish_failure_retains_jobs_until_month_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cds_client: FakeCDSClient,
):
    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    publish = era5.DownloadCache.publish

    def fail_publish(_self, _paths):  # noqa: ANN001
        msg = 'cache upload interrupted'
        raise OSError(msg)

    monkeypatch.setattr(era5.DownloadCache, 'publish', fail_publish)
    store = get_store(tmp_path / 'store')
    bbox = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(OSError, match='cache upload interrupted'):
        era5.download_era5_data(store, bbox=bbox)
    attrs = era5.group_attrs(store, '')
    assert 'last_month' not in attrs
    assert set(attrs['era5_requests'].values()) == {'job-1', 'job-2', 'job-3', 'job-4'}
    assert era5.last_date(store, 'daily') is None
    monkeypatch.setattr(era5.DownloadCache, 'publish', publish)
    # The temporary raw directory disappears: retry recovers the first job.
    era5.download_era5_data(get_store(tmp_path / 'store'), bbox=bbox)
    assert len(cds_client.calls) == 6
    assert set(cds_client.resumed) == {'job-1', 'job-2', 'job-3', 'job-4'}
    assert era5.group_attrs(store, '')['era5_requests'] == {}
    assert era5.available_period(store) == (date(2000, 1, 1), date(2000, 1, 31))


def test_s3_monthly_cache_reused_by_fresh_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cds_client: FakeCDSClient
):
    from fsspec.implementations.memory import MemoryFileSystem

    from weatheasy.file_storage import FileStore

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    fs = MemoryFileSystem()
    cache = FileStore(fs, f'{tmp_path.name}/raw')
    bbox = (55.3, 37.1, 55.2, 37.3)
    era5.download_era5_data(get_store(tmp_path / 'first'), cache, bbox=bbox)
    assert len(cds_client.calls) == 6
    assert len(cache.child('era5').glob('*.grib')) == 6
    era5.download_era5_data(get_store(tmp_path / 'second'), cache, bbox=bbox)
    assert len(cds_client.calls) == 6
    with era5.open_dataset(get_store(tmp_path / 'second'), 'daily') as ds:
        assert ds.sizes['time'] == 31


def test_grib_daily_reduction_is_lazy_and_uses_next_midnight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from collections import Counter

    from cfgrib import dataset as grib_dataset

    decoded = Counter()
    decode = grib_dataset.get_values_in_order

    def decode_field(message, shape):  # noqa: ANN001
        decoded[message['cfVarName']] += 1
        assert shape == (2, 3)
        return decode(message, shape)

    monkeypatch.setattr(grib_dataset, 'get_values_in_order', decode_field)
    main = read_grib(FIXTURES / 'day.grib', set(REQUIRED_INPUTS))
    boundary = read_grib(FIXTURES / 'midnight.grib', set(REQUIRED_INPUTS))
    assert not decoded  # Opening/merging fields must not unpack hourly weather arrays.
    np.testing.assert_allclose(main.latitude, [55.3, 55.2])
    np.testing.assert_allclose(main.longitude, [37.1, 37.2, 37.3])
    assert np.isnan(main.t2m.isel(time=0).to_numpy()[0, 0])
    assert decoded == {'t2m': 1}  # Select the hour before materializing the field.
    decoded.clear()
    result = era5.process_day(main.isel(time=slice(0, 24)), boundary, date(2000, 1, 1))
    for name in ('t2m', 'd2m', 'sp', 'u10', 'v10', 'skt', 'swvl1'):
        assert decoded[name] == 24
    assert decoded['tp'] == 1
    for name, expected in {
        'WIND': 5,
        'UGRD': 0,
        'VGRD': 0,
        'PRES': 1000,
        'PEVPR': 4,
        'LHTFL': 100,
        'SHTFL': 50,
        'DSWRF': 100,
        'DLWRF': 200,
        'USWRF': 25,
        'ULWRF': 210,
        'WEASD': 20,
    }.items():
        np.testing.assert_allclose(result[name], expected, atol=1e-4)
    np.testing.assert_allclose(result.PRATE * 86400, 12, atol=1e-4)
    assert np.isnan(result.TMP[0, 1, 0])
    assert np.isnan(result.TMIN[0, 1, 0])
    assert result.spatial_ref.attrs['semi_major_axis'] == 6371229
    assert result.attrs['source_grib']['main']['t2m'] == {'edition': 2, 'radius': 6367470}
    assert result.attrs['source_grib']['boundary']['tp'] == {'edition': 2, 'radius': 6367470}
    store = get_store(tmp_path / 'daily.zarr')
    era5.write_batches(store, 'daily', [result])
    with open_dataset(store, 'daily') as stored:
        assert stored.attrs['source_grib'] == result.attrs['source_grib']
        assert stored.rio.crs.to_dict()['R'] == 6371229


@pytest.mark.parametrize('remote', [False, True])
def test_processing_failure_preserves_download_before_grib_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cds_client: FakeCDSClient, *, remote: bool
):
    from fsspec.implementations.memory import MemoryFileSystem

    from weatheasy.file_storage import FileStore

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    backing = FileStore(MemoryFileSystem(), f'{tmp_path.name}/raw')
    location = backing if remote else tmp_path / 'cache'
    read = era5.read_grib
    reads = []
    remote_reads = []
    original_open = backing.fs.open

    def track_open(path, mode='rb', **kwargs):  # noqa: ANN001
        if mode == 'rb':
            remote_reads.append(path)
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(backing.fs, 'open', track_open)

    def fail_read(path, _required):  # noqa: ANN001
        reads.append(path)
        if remote:
            assert backing.child('era5').exists(path.name)
            old = time.time() - 30 * 86400
            os.utime(path, (old, old))
            obsolete = path.parent / 'obsolete.grib'
            obsolete.write_bytes(b'obsolete')
            os.utime(obsolete, (old, old))
        else:
            assert path.parent == tmp_path / 'cache' / 'era5'
            assert path.is_file()
        msg = 'GRIB processing interrupted'
        raise ValueError(msg)

    monkeypatch.setattr(era5, 'read_grib', fail_read)
    store = get_store(tmp_path / 'store')
    bbox = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(ValueError, match='GRIB processing interrupted'):
        era5.download_era5_data(store, location, bbox=bbox)
    assert len(reads) == 1
    assert reads[0].is_file()
    assert len(cds_client.calls) == 4
    assert era5.last_date(store, 'daily') is None
    monkeypatch.setattr(era5, 'read_grib', read)
    era5.download_era5_data(get_store(tmp_path / 'store'), location, bbox=bbox)
    if remote:
        assert remote_reads == []
        assert not reads[0].exists()
        assert not (reads[0].parent / 'obsolete.grib').exists()
    assert len(cds_client.calls) == 6
    assert set(cds_client.resumed) == {'job-2', 'job-3', 'job-4'}
    assert era5.available_period(store) == (date(2000, 1, 1), date(2000, 1, 31))


@pytest.mark.parametrize('limit', [1, 3])
def test_submission_window_and_restart_keep_next_month_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cds_client: FakeCDSClient, limit: int
):
    downloaded = []

    def download(_self, _request, call, path):  # noqa: ANN001
        downloaded.append(call)
        Path(path).write_bytes(b'GRIB')
        # The oldest job finishes after a newer result has been downloaded.
        cds_client.jobs['job-1'].results_ready = True

    monkeypatch.setattr(FakeCDSClient, '_download', download)
    monkeypatch.setattr(
        era5._Downloader, '_read', lambda _self, part: requested_source(part.month, part.request)
    )
    store = get_store(tmp_path / 'store')
    cache = tmp_path / 'cache'
    bbox = (55.3, 37.1, 55.2, 37.3)
    loader = era5._Downloader(cds_client, cache, bbox, store, limit)
    loader.plan(date(2000, 1, 1), date(2000, 3, 10))
    assert len(cds_client.calls) == limit
    if limit == 3:
        cds_client.jobs['job-1'].results_ready = False
    loader.retrieve_month(date(2000, 1, 1))
    assert downloaded == ([1, 2, 3, 4, 5, 6] if limit == 1 else [2, 1, 3, 4, 5, 6])
    assert len(cds_client.calls) == 6 + limit
    loader.commit(date(2000, 1, 1))
    saved = era5.group_attrs(store, '')['era5_requests']
    assert set(saved.values()) == {f'job-{i}' for i in range(7, 7 + limit)}
    restarted = era5._Downloader(cds_client, cache, bbox, store, limit)
    restarted.plan(date(2000, 2, 1), date(2000, 3, 10))
    assert len(cds_client.calls) == 6 + limit
    assert set(cds_client.resumed) == set(saved.values())
