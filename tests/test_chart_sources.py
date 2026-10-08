import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


CHART = Path(__file__).resolve().parents[1] / 'charts' / 'weatheasy'


def render(tmp_path: Path, values: dict) -> subprocess.CompletedProcess[str]:
    helm = shutil.which('helm')
    if helm is None:
        pytest.skip('Helm is not installed')
    config = tmp_path / 'values.yaml'
    config.write_text(yaml.safe_dump(values))
    return subprocess.run(  # noqa: S603
        [helm, 'template', 'test', str(CHART), '-f', str(config)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_era5_only_chart_connects_one_store_without_jobs(tmp_path: Path):
    result = render(tmp_path, {'sources': {'era5': {'store': 's3://weather/era5.zarr'}}})
    assert result.returncode == 0, result.stderr
    manifests = list(yaml.safe_load_all(result.stdout))
    assert not any(m['kind'] == 'CronJob' for m in manifests)
    api = next(m for m in manifests if m['kind'] == 'Deployment')
    env = api['spec']['template']['spec']['containers'][0]['env']
    stores = {v['name']: v['value'] for v in env if v['name'].startswith('WEATHEASY__SOURCES__')}
    assert stores == {'WEATHEASY__SOURCES__ERA5__STORE': 's3://weather/era5.zarr'}
    assert not api['spec']['template']['spec']['containers'][0].get('envFrom')


@pytest.mark.parametrize('cache', [None, 's3://weather/raw', '/data/raw'])
def test_update_jobs_use_their_own_stores_and_credentials(tmp_path: Path, cache: str | None):
    values = {
        'sources': {
            'cfs2': {
                'store': 's3://cfs/weather.zarr',
                'update': {'enabled': True},
            },
            'era5': {
                'store': 's3://era/weather.zarr',
                'update': {
                    'enabled': True,
                    'envFrom': [{'secretRef': {'name': 'cds'}}],
                    'correction': {
                        'enabled': True,
                        'dem': 's3://dem/grid.zarr',
                        'artifacts': 's3://models/era5',
                    },
                },
            },
            'cmip6': {'store': 's3://climate/weather.zarr'},
        }
    }
    if cache is not None:
        values['downloadCache'] = cache
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    jobs = [m for m in yaml.safe_load_all(result.stdout) if m['kind'] == 'CronJob']
    assert {j['metadata']['name'] for j in jobs} == {'test-weatheasy-cfs2', 'test-weatheasy-era5'}
    for job in jobs:
        source = job['metadata']['name'].rsplit('-', 1)[1]
        spec = job['spec']['jobTemplate']['spec']['template']['spec']
        containers = spec.get('initContainers', []) + spec['containers']
        assert len(containers) == (2 if source == 'era5' else 1)
        for container in containers:
            args = container['args']
            assert args[args.index('--store') + 1] == values['sources'][source]['store']
            if container['name'] == 'download':
                assert args[args.index('--download-cache') + 1] == (
                    cache if cache is not None else '/var/lib/weatheasy/downloads'
                )
            assert container.get('envFrom', []) == (
                [{'secretRef': {'name': 'cds'}}] if source == 'era5' else []
            )


@pytest.mark.parametrize(
    'values',
    [
        {'sources': {'era5': {'update': {'enabled': True}}}},
        {'sources': {}},
    ],
)
def test_chart_rejects_missing_required_store(tmp_path: Path, values: dict):
    result = render(tmp_path, values)
    assert result.returncode != 0
    assert 'values' in result.stderr


def test_update_only_chart_needs_no_other_sources(tmp_path: Path):
    result = render(
        tmp_path,
        {
            'api': {'enabled': False},
            'sources': {
                'cfs2': {'store': 's3://weather/cfs2.zarr', 'update': {'enabled': True}},
            },
        },
    )
    assert result.returncode == 0, result.stderr
    manifests = list(yaml.safe_load_all(result.stdout))
    assert not any(m['kind'] == 'Deployment' for m in manifests)
    assert len([m for m in manifests if m['kind'] == 'CronJob']) == 1
