from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from weatheasy import Coords, const, era5, get_era5_data
from weatheasy.storage import StorageError, open_dataset
from weatheasy.util import get_storage


def source(month, *, boundary=False):
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
    )


@pytest.mark.parametrize('month', [date(2000, 2, 1), date(2000, 12, 1)])
def test_daily_units_midnight_wind_and_coordinates(month):
    main = source(month)
    boundary = source(era5.next_month(month), boundary=True)
    # 23 UTC is deliberately wrong; only the NEXT midnight carries full accumulation.
    main['tp'][23::24] = 0.001
    boundary['tp'][:] = 0.012
    main['t2m'][0, 0, 0] = np.nan
    result = era5.process_month(main, boundary, month)
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
    np.testing.assert_allclose(result.latitude, [55.2, 55.3])


@pytest.mark.parametrize(
    'name', sorted({v.nc_name for v in const.ERA5_VARS.values()} | set(era5.EXTRA))
)
def test_all_inputs_required(name):
    month = date(2000, 1, 1)
    with pytest.raises(StorageError, match='Missing ERA5'):
        era5.process_month(
            source(month).drop_vars(name), source(era5.next_month(month), boundary=True), month
        )


def test_boundary_and_timestamp_validation():
    month = date(2000, 12, 1)
    main = source(month)
    boundary = source(date(2001, 1, 1), boundary=True)
    with pytest.raises(StorageError, match='timestamps'):
        era5.process_month(main, boundary.isel(time=slice(0, 0)), month)
    bad = main.assign_coords(time=np.roll(main.time.to_numpy(), 1))
    with pytest.raises(StorageError, match='timestamps'):
        era5.process_month(bad, boundary, month)
    with pytest.raises(StorageError, match='coordinates'):
        era5.process_month(main, boundary.assign_coords(longitude=[38.1, 38.2, 38.3]), month)


def test_public_monthly_download_resume_getter_and_cli(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    calls = []

    class Client:
        def retrieve(self, _dataset, request):
            assert len(request['variable']) <= 8
            assert (
                len(request['variable']) * len(request['day']) * len(request['time']) * 2 <= 12000
            )
            calls.append(request)
            month = date(int(request['year']), int(request['month']), 1)
            ds = requested_source(month, request)
            return SimpleNamespace(
                download=lambda path: ds.rename({'time': 'valid_time'}).to_netcdf(path)
            )

    monkeypatch.setitem(sys.modules, 'cdsapi', SimpleNamespace(Client=Client))
    root = get_storage(str(tmp_path / 'store'))
    area = (55.3, 37.1, 55.2, 37.3)
    era5.download_era5_data(root, tmp_path / 'cache', area=area)
    assert len(calls) == 6
    assert root['era5'].attrs['last_month'] == [2000, 1]
    root['era5'].attrs['last_month'] = [1999, 12]  # crash before convenience marker
    era5.download_era5_data(root, tmp_path / 'cache', area=area)
    assert len(calls) == 6
    assert root['era5'].attrs['last_month'] == [2000, 1]
    values = get_era5_data(
        root=root,
        begin=date(2000, 1, 31),
        end=date(2000, 1, 31),
        coords=Coords(55.3, 37.2),
        variables=['WIND', 'TMP'],
    )
    np.testing.assert_allclose(values, [[5], [20]], atol=1e-5)
    before = dict(root['era5'].attrs)
    with pytest.raises(StorageError, match='different requested area'):
        era5.download_era5_data(root, area=(55.3, 38.1, 55.2, 38.3))
    assert dict(root['era5'].attrs) == before
    with open_dataset(root.store, 'era5/daily') as ds:
        assert ds.spatial_ref.attrs['semi_major_axis'] == 6367470
        assert ds.WIND.attrs['units'] == 'm/s'
    from weatheasy.__main__ import main

    output = tmp_path / 'result.csv'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'weatheasy',
            '-d',
            str(tmp_path / 'store'),
            '-o',
            str(output),
            'era5',
            '2000-01-31',
            '2000-01-31',
            '55.3',
            '37.2',
            'WIND',
        ],
    )
    main(configure_logging=False)
    assert output.read_text() == 'DATE,WIND\n2000-01-31,5\n'


def test_failed_month_and_append_recover(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 3, 10, tzinfo=UTC))
    monkeypatch.setitem(sys.modules, 'cdsapi', SimpleNamespace(Client=lambda: None))
    broken = True

    def retrieve(_client, _cache, month, _area, *, boundary):
        ds = source(month, boundary=boundary)
        return ds.isel(time=slice(0, 0)) if broken and month.month == 3 else ds

    monkeypatch.setattr(era5, '_retrieve', retrieve)
    root = get_storage(str(tmp_path / 'store'))
    area = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(StorageError, match='timestamps'):
        era5.download_era5_data(root, area=area)
    assert root['era5'].attrs['last_month'] == [2000, 1]
    assert era5.available_period(root)[1] == date(2000, 1, 31)
    broken = False
    original = xr.Dataset.to_zarr

    def fail_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        msg = 'interrupted after appending arrays'
        raise OSError(msg)

    monkeypatch.setattr(xr.Dataset, 'to_zarr', fail_write)
    with pytest.raises(OSError, match='interrupted'):
        era5.download_era5_data(root, area=area)
    assert root['era5'].attrs['last_month'] == [2000, 1]
    monkeypatch.setattr(xr.Dataset, 'to_zarr', original)
    era5.download_era5_data(root, area=area)
    assert era5.available_period(root) == (date(2000, 1, 1), date(2000, 2, 29))
    assert root['era5/daily'].attrs['committed_length'] == 60


def test_cache_includes_area(tmp_path):
    calls = []

    class Client:
        def retrieve(self, _dataset, request):
            assert len(request['variable']) <= 8
            assert (
                len(request['variable']) * len(request['day']) * len(request['time']) * 2 <= 12000
            )
            calls.append(request)
            return SimpleNamespace(
                download=lambda path: requested_source(date(2000, 1, 1), request).to_netcdf(path)
            )

    for area in [(55.3, 37.1, 55.2, 37.3), (55.3, 38.1, 55.2, 38.3)]:
        era5._retrieve(Client(), tmp_path, date(2000, 1, 1), area, boundary=False)  # noqa: SLF001
    assert len(calls) == 6
    assert len(list(tmp_path.glob('*.nc'))) == 6


def test_actual_axes_reject_different_region():
    with pytest.raises(StorageError, match='longitude'):
        era5._validate_area(source(date(2000, 1, 1)), (55.3, 38.1, 55.2, 38.3))  # noqa: SLF001


def test_split_netcdf_response_preserves_explicit_crs(tmp_path):
    import zipfile

    month = date(2000, 1, 1)
    ds = source(month).rio.write_crs('EPSG:4326')
    files = [tmp_path / 'instant.nc', tmp_path / 'accumulated.nc']
    accumulated = {v.nc_name for v in const.ERA5_VARS.values() if v.accumulated} | {'ssr', 'str'}
    ds[sorted(set(ds.data_vars) - accumulated)].to_netcdf(files[0])
    ds[sorted(accumulated)].to_netcdf(files[1])
    archive_path = tmp_path / 'response.nc'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for file in files:
            archive.write(file, file.name)
    restored = era5._read_response(archive_path)  # noqa: SLF001
    boundary = source(era5.next_month(month), boundary=True).rio.write_crs('EPSG:4326')
    result = era5.process_month(restored, boundary, month)
    assert result.rio.crs.to_epsg() == 4326


def test_download_cli_dispatch(monkeypatch, tmp_path):
    import sys

    from weatheasy import download

    calls = []
    monkeypatch.setattr(
        download, 'download_era5_data', lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'weatheasy.download',
            '-d',
            str(tmp_path / 'store'),
            'era5',
            '--area',
            '55.3',
            '37.1',
            '55.2',
            '37.3',
        ],
    )
    download.main(configure_logging=False)
    assert calls[0][1]['area'] == (55.3, 37.1, 55.2, 37.3)


def requested_source(month, request):
    names = {v.cds_name: v.nc_name for v in const.ERA5_VARS.values()}
    names.update({cds: nc for nc, cds in era5.EXTRA.items()})
    return source(month, boundary=len(request['time']) == 1)[
        [names[name] for name in request['variable']]
    ]


def test_partial_request_missing_variable_never_publishes(tmp_path, monkeypatch):
    import sys

    import zarr

    monkeypatch.setattr(const, 'ERA5_FIRST_YEAR', 2000)
    monkeypatch.setattr(era5, 'utc_now', lambda: datetime(2000, 2, 10, tzinfo=UTC))
    broken = True
    calls = []

    class Client:
        def retrieve(self, _dataset, request):
            assert len(request['variable']) <= 8
            calls.append(request)
            month = date(int(request['year']), int(request['month']), 1)
            ds = requested_source(month, request)
            if broken and len(calls) == 2:
                ds = ds.drop_vars(next(iter(ds.data_vars)))
            return SimpleNamespace(download=lambda path: ds.to_netcdf(path))

    monkeypatch.setitem(sys.modules, 'cdsapi', SimpleNamespace(Client=Client))
    root = get_storage(str(tmp_path / 'store'))
    area = (55.3, 37.1, 55.2, 37.3)
    with pytest.raises(StorageError, match='Missing ERA5'):
        era5.download_era5_data(root, tmp_path / 'cache', area=area)
    assert 'last_month' not in root['era5'].attrs
    assert 'daily' not in root['era5']
    assert len(list((tmp_path / 'cache/era5').glob('*.nc'))) == 1
    broken = False
    era5.download_era5_data(root, tmp_path / 'cache', area=area)
    assert len(calls) == 7  # six successful parts; first reused from cache after failure
    readonly = zarr.open_group(str(tmp_path / 'store'), mode='r')
    result = get_era5_data(
        root=readonly,
        begin=date(2000, 1, 1),
        end=date(2000, 1, 1),
        coords=Coords(55.3, 37.2),
        variables=['WIND'],
    )
    np.testing.assert_allclose(result, [[5]])
