"""HTTP DEM preparation reads real metadata/chunks using the selected credentials."""

import json
from collections.abc import Callable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.abc.store import Store

from weatheasy._cli import main
from weatheasy._cli._storage import Storage
from weatheasy.correction import terrain
from weatheasy.storage import get_store, prepare_dataset, write_batch


type HTTPDem = tuple[str, list[tuple[str, dict[str, str]]], dict[str, str], Path]


@pytest.fixture(name='http_dem')
def create_http_dem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lats = np.arange(53.75, 57, 0.25)
    lons = np.arange(35.75, 39, 0.25)
    ds = xr.Dataset(
        {'dsm': (('lat', 'lon'), np.broadcast_to(100 + 8 * (lons - 37), (13, 13)))},
        coords={'lat': lats, 'lon': lons},
    )
    ds.to_zarr(
        tmp_path / 'dem.zarr',
        zarr_format=2,
        consolidated=True,
        encoding={'dsm': {'chunks': (5, 5)}},
    )
    netrc = tmp_path / 'netrc'
    netrc.write_text('')
    monkeypatch.setenv('NETRC', str(netrc))
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    requests = []
    required = {}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            if any(self.headers.get(key) != value for key, value in required.items()):
                self.send_error(401)
            elif self.path.endswith('/'):
                # Object stores need not support directory enumeration.
                self.send_error(404)
            else:
                super().do_GET()

        def log_message(self, format: str, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(tmp_path)))
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/dem.zarr', requests, required, netrc
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(name='native_source')
def create_native_source(tmp_path: Path):
    def create(crs: str | None):
        ds = xr.Dataset(
            {'elevation': (('y', 'x'), np.array([[10.0, 20.0], [30.0, 40.0]]))},
            coords={'x': [37.0, 38.0], 'y': [55.0, 56.0]},
        )
        if crs is not None:
            ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs(crs)
        store = get_store(tmp_path / 'dem.zarr')
        ds.to_zarr(store, consolidated=True, zarr_format=3)
        return store

    return create


@pytest.mark.parametrize(
    ('command', 'auth'),
    [
        ('prepare-dem', auth)
        for auth in ('public', 'netrc', 'bearer', 'custom-netrc', 'basic-header')
    ]
    + [('prepare-stations', 'bearer')],
)
def test_http_preparation_auth_and_consolidated_reads(
    tmp_path: Path, http_dem: HTTPDem, command: str, auth: str, caplog: pytest.LogCaptureFixture
):
    url, requests, required, netrc = http_dem
    arguments = ['correction', command]
    if command == 'prepare-dem':
        weather = get_store(tmp_path / 'weather.zarr')
        ds = xr.Dataset(
            {
                'TMP': (
                    ('time', 'latitude', 'longitude'),
                    np.ones((1, 3, 3), np.float32),
                    {'units': 'degC'},
                )
            },
            coords={
                'time': [np.datetime64('2020-01-01')],
                'latitude': [55.0, 55.5, 56.0],
                'longitude': [37.0, 37.5, 38.0],
            },
        )
        write_batch(weather, 'daily', prepare_dataset(ds, crs='EPSG:4326'))
        output = tmp_path / 'terrain.zarr'
        arguments += ['era5', '--store', str(tmp_path / 'weather.zarr')]
    else:
        stations = tmp_path / 'stations.parquet'
        pd.DataFrame({'lat': [55.0], 'lon': [37.5]}).to_parquet(stations)
        output = tmp_path / 'prepared.parquet'
        arguments += ['--stations', str(stations)]
    arguments += [
        '--dem-source',
        url,
        '--dem-elevation',
        'dsm',
        '--dem-x',
        'lon',
        '--dem-y',
        'lat',
        '--output',
        str(output),
    ]
    if auth != 'public':
        # A matching netrc proves explicit Authorization takes precedence.
        netrc.write_text('machine 127.0.0.1 login edh password test-key\n')
        required['Authorization'] = 'Basic ZWRoOnRlc3Qta2V5'
    if auth in ('bearer', 'custom-netrc', 'basic-header'):
        if auth == 'bearer':
            headers = {'authorization': 'Bearer test-token'}
            required['Authorization'] = 'Bearer test-token'
        elif auth == 'basic-header':
            headers = {'Authorization': 'Basic ZWRoOnRlc3Qta2V5'}
        else:
            headers = {'X-API-Key': 'test-custom-key'}
            required['X-API-Key'] = 'test-custom-key'
        headers_file = tmp_path / 'headers.json'
        headers_file.write_text(json.dumps(headers))
        arguments += ['--dem-headers-file', str(headers_file)]
    caplog.set_level('DEBUG')
    main(arguments, configure_logging=False)
    secrets = ('test-token', 'test-custom-key', 'test-key', *required.values())
    assert not any(secret in caplog.text for secret in secrets)
    if command == 'prepare-dem':
        with xr.open_zarr(output, chunks=None, consolidated=True) as result:
            np.testing.assert_allclose(result.elevation, [[100, 104, 108]] * 3)
    else:
        assert pd.read_parquet(output).elevation.tolist() == [104.0]
    assert requests
    assert any(path == '/dem.zarr/.zmetadata' for path, _ in requests)
    assert any(path.startswith('/dem.zarr/dsm/') for path, _ in requests)
    assert not any(path.endswith('/') for path, _ in requests)
    for _, received in requests:
        normalized = {key.lower(): value for key, value in received.items()}
        for key, value in required.items():
            assert normalized.get(key.lower()) == value
        if auth == 'public':
            assert 'authorization' not in normalized


@pytest.mark.parametrize(
    'contents',
    [
        '[]',
        '{"Authorization": 123}',
        '{"Authorization": "sensitive-secret",}',
        '{"Authorization": "sensitive-secret\\n"}',
    ],
)
def test_invalid_header_file_does_not_disclose_contents(tmp_path: Path, contents: str):
    headers_file = tmp_path / 'headers.json'
    headers_file.write_text(contents)
    with pytest.raises(ValueError, match='JSON object of string headers') as exc:
        Storage().dem_source('https://example.test/dem.zarr', headers_file=str(headers_file))
    assert 'sensitive-secret' not in str(exc.value)


def test_headers_are_not_applied_to_local_source(tmp_path: Path):
    with pytest.raises(ValueError, match='requires an HTTP or HTTPS'):
        Storage().dem_source(str(tmp_path / 'dem.zarr'), headers_file='headers.json')


@pytest.mark.parametrize('crs', [None, 'EPSG:4326'])
def test_source_uses_wgs84_with_or_without_metadata(
    native_source: Callable[[str | None], Store], crs: str | None
):
    with terrain._native(native_source(crs), 'elevation', 'x', 'y') as field:
        np.testing.assert_array_equal(field.values, [[10.0, 20.0], [30.0, 40.0]])
        assert field.rio.crs == 'EPSG:4326'


def test_source_rejects_non_geographic_crs(native_source: Callable[[str | None], Store]):
    with (
        pytest.raises(ValueError, match='requires EPSG:4326'),
        terrain._native(native_source('EPSG:3857'), 'elevation', 'x', 'y'),
    ):
        pass
