import asyncio
import io
import json
import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from weatheasy import Coords, get_cfs2_data, get_cmip6_data, get_era5_data
from weatheasy.storage import open_dataset, prepare_dataset, write_batch
from weatheasy.util import float_formatter_factory, get_storage


BEGIN, END = date(2020, 2, 28), date(2020, 3, 1)


def weather_store(tmp_path, source):
    root = get_storage(str(tmp_path / source))
    name = 'pr' if source == 'cmip6' else 'PRATE'
    scale = 21600 if source == 'cfs2' else 86400
    raw = np.array([8, np.nan, 2], dtype=np.float32) / scale
    dataset = prepare_dataset(
        xr.Dataset(
            {
                name: (
                    ('time', 'latitude', 'longitude'),
                    np.broadcast_to(raw[:, None, None], (3, 2, 2)).copy(),
                    {'units': 'kg m-2 s-1'},
                )
            },
            coords={
                'time': np.arange('2020-02-28', '2020-03-02', dtype='datetime64[D]'),
                'latitude': [54.0, 55.0],
                'longitude': [36.0, 37.0],
            },
        ),
        crs='EPSG:4326',
    )
    if source == 'cfs2':
        write_batch(root.store, 'cfs2/reanalysis/flx', dataset.isel(time=slice(0, 2)))
        write_batch(root.store, 'cfs2/forecast/flx', dataset.isel(time=slice(2, None)))
        root['cfs2'].attrs['updated'] = '2020-03-04'  # forecast begins March 1
        path = 'cfs2/reanalysis/flx'
    else:
        path = 'era5/daily' if source == 'era5' else 'cmip6/pr'
        write_batch(root.store, path, dataset)
    return root, name, path, raw


@pytest.mark.parametrize('source', ['cfs2', 'era5', 'cmip6'])
def test_daily_getter_and_csv_units(tmp_path, monkeypatch, source):
    from weatheasy.__main__ import main

    root, name, path, raw = weather_store(tmp_path, source)
    getter = {'cfs2': get_cfs2_data, 'era5': get_era5_data, 'cmip6': get_cmip6_data}[source]
    result = getter(root=root, begin=BEGIN, end=END, coords=Coords(55, 37), variables=[name])
    np.testing.assert_allclose(result, [[8, np.nan, 2]])
    # Retrieval leaves native source values and metadata intact.
    with open_dataset(root.store, path) as ds:
        np.testing.assert_allclose(ds[name][:, 0, 0], raw[: ds.sizes['time']])
        assert ds[name].attrs['units'] == 'kg m-2 s-1'
    target = tmp_path / 'result.csv'
    argv = [
        'weatheasy',
        '-d',
        str(tmp_path / source),
        '-o',
        str(target),
        source,
        BEGIN.isoformat(),
        END.isoformat(),
        '55',
        '37',
        name,
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    main(configure_logging=False)
    expected = f'DATE,{name}\n2020-02-28,8\n2020-02-29,NA\n2020-03-01,2\n'
    assert target.read_text() == expected

    class Output(io.StringIO):
        def close(self):
            pass

    stdout = Output()
    monkeypatch.setattr(sys, 'stdout', stdout)
    argv[4] = 'stdout'
    main(configure_logging=False)
    assert stdout.getvalue() == expected


@pytest.mark.parametrize('source', ['cmip6', 'cfs2'])
def test_http_precipitation_units_and_response_shape(tmp_path, monkeypatch, source):
    pytest.importorskip('fastapi')
    from weatheasy.web import controller

    root, name, _, _ = weather_store(tmp_path, source)
    getter = get_cmip6_data if source == 'cmip6' else get_cfs2_data
    monkeypatch.setattr(
        controller,
        'get_config',
        lambda: SimpleNamespace(storage=root, format_float=float_formatter_factory('null', 6)),
    )

    async def query():
        response = await controller.get_data(
            getter, {'begin': BEGIN, 'end': END, 'coords': Coords(55, 37), 'variables': [name]}
        )
        return json.loads(''.join([chunk async for chunk in response.body_iterator]))

    assert asyncio.run(query()) == [
        {name: 8, 'date': '2020-02-28'},
        {name: None, 'date': '2020-02-29'},
        {name: 2, 'date': '2020-03-01'},
    ]
    variables = controller.get_variables()
    info = getattr(variables, source)[name]
    assert 'mm/day' in info.en


@pytest.mark.parametrize('source', ['cfs2', 'era5'])
def test_raw_and_corrected_precipitation_have_same_units(tmp_path, source):
    root, _, path, _ = weather_store(tmp_path, source)
    with open_dataset(root.store, path) as raw:
        corrected = raw.rename({'PRATE': 'PRATE_corr'}).load()
    corrected['PRATE_corr'][:] = 12
    corrected.PRATE_corr.attrs['units'] = 'mm/day'
    correction_path = 'cfs2/reanalysis/correction' if source == 'cfs2' else 'era5/correction'
    write_batch(root.store, correction_path, corrected)
    getter = get_cfs2_data if source == 'cfs2' else get_era5_data
    values = getter(
        root=root,
        begin=BEGIN,
        end=BEGIN,
        coords=Coords(55, 37),
        variables=['PRATE_corr', 'PRATE'],
    )
    np.testing.assert_allclose(values, [[12], [8]])
