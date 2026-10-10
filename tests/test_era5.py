import sys
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
    def read_response(path, _expected, _required):  # noqa: ANN001
        with xr.open_dataset(path, engine='netcdf4', decode_coords='all') as ds:
            return ds.load()

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
    assert result.spatial_ref.attrs['semi_major_axis'] == 6367470
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


@pytest.mark.parametrize('name', REQUIRED_INPUTS)
def test_all_inputs_required(name: str):
    month = date(2000, 1, 1)
    main = source(month)
    assert name in main
    with pytest.raises(StorageError, match='Missing ERA5'):
        era5._validate(main.drop_vars(name), month, boundary=False)


def test_boundary_and_timestamp_validation():
    month = date(2000, 12, 1)
    main = source(month)
    boundary = source(date(2001, 1, 1), boundary=True)
    with pytest.raises(StorageError, match='timestamps'):
        era5._validate(boundary.isel(time=slice(0, 0)), date(2001, 1, 1), boundary=True)
    bad = main.assign_coords(time=np.roll(main.time.to_numpy(), 1))
    with pytest.raises(StorageError, match='timestamps'):
        era5._validate(bad, month, boundary=False)
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
        assert ds.spatial_ref.attrs['semi_major_axis'] == 6367470
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
    with pytest.raises(StorageError, match='Missing ERA5'):
        era5.download_era5_data(root.store, bbox=bbox)
    assert root[''].attrs['last_month'] == [2000, 1]
    assert era5.available_period(root.store)[1] == date(2000, 1, 31)
    assert len(root[''].attrs['era5_requests']) == 4
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
        era5._Downloader(cds_client, tmp_path / 'cache', bbox, get_store(store_path)).retrieve(
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
    restarted = era5._Downloader(cds_client, tmp_path / 'cache', bbox, get_store(store_path))
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


def test_actual_axes_reject_different_region():
    with pytest.raises(StorageError, match='longitude'):
        era5._validate_bbox(source(date(2000, 1, 1)), (55.3, 38.1, 55.2, 38.3))


def test_grib_single_midnight():
    ds = read_grib(
        FIXTURES / 'midnight.grib',
        np.array(['2000-01-02'], dtype='datetime64[h]'),
        set(REQUIRED_INPUTS),
    )
    assert ds.sizes['time'] == 1
    np.testing.assert_allclose(ds.tp, 0.012, atol=1e-7)
    assert ds.rio.crs.to_dict()['R'] == 6367470


@pytest.mark.parametrize(
    ('invalid', 'error'),
    [('duplicate', 'Duplicate'), ('unexpected', 'Unexpected'), ('missing_hour', 'timestamps')],
)
def test_grib_rejects_invalid_response(tmp_path: Path, invalid: str, error: str):
    contents = (FIXTURES / 'day.grib').read_bytes()
    path = tmp_path / 'response.grib'
    path.write_bytes(contents * 2 if invalid == 'duplicate' else contents)
    expected = np.arange(
        np.datetime64('2000-01-01T00'), np.datetime64('2000-01-02T01'), np.timedelta64(1, 'h')
    )
    required = set(REQUIRED_INPUTS)
    if invalid == 'unexpected':
        required.remove('sp')
    elif invalid == 'missing_hour':
        expected = np.append(expected, np.datetime64('2000-01-02T01'))
    with pytest.raises(StorageError, match=error):
        read_grib(path, expected, required)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['response.grib']


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


def requested_source(month: date, request: Mapping):
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
    names.update({cds: nc for nc, cds in era5.EXTRA.items()})
    return source(month, boundary=len(request['time']) == 1)[
        [names[name] for name in request['variable']]
    ]


def test_partial_request_missing_variable_never_publishes(
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
    root = get_storage(str(tmp_path / 'store'))
    bbox = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(StorageError, match='Missing ERA5'):
        era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert 'last_month' not in root[''].attrs
    assert 'daily' not in root
    assert len(list((tmp_path / 'cache/era5').glob('*.grib'))) == 1
    broken = False
    era5.download_era5_data(root.store, tmp_path / 'cache', bbox=bbox)
    assert len(cds_client.calls) == 6  # Reuse the interrupted job as well as the cached file.
    assert cds_client.resumed == ['job-2']
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
    assert set(attrs['era5_requests'].values()) == {f'job-{n}' for n in range(1, 7)}
    assert era5.last_date(store, 'daily') is None
    monkeypatch.setattr(era5.DownloadCache, 'publish', publish)
    # Both temporary raw directories disappear: the retry must recover all six jobs.
    era5.download_era5_data(get_store(tmp_path / 'store'), bbox=bbox)
    assert len(cds_client.calls) == 6
    assert set(cds_client.resumed) == {f'job-{n}' for n in range(1, 7)}
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


def test_grib_daily_reduction_is_lazy_and_uses_next_midnight(monkeypatch: pytest.MonkeyPatch):
    from collections import Counter

    from cfgrib import dataset as grib_dataset

    decoded = Counter()
    decode = grib_dataset.get_values_in_order

    def decode_field(message, shape):  # noqa: ANN001
        decoded[message['cfVarName']] += 1
        assert shape == (2, 3)
        return decode(message, shape)

    monkeypatch.setattr(grib_dataset, 'get_values_in_order', decode_field)
    expected = np.arange(
        np.datetime64('2000-01-01T00'), np.datetime64('2000-01-02T01'), np.timedelta64(1, 'h')
    )
    main = read_grib(FIXTURES / 'day.grib', expected, set(REQUIRED_INPUTS))
    boundary = read_grib(FIXTURES / 'midnight.grib', expected[-1:], set(REQUIRED_INPUTS))
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
    assert result.spatial_ref.attrs['semi_major_axis'] == 6367470
