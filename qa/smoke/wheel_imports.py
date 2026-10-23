"""Require every workspace distribution to load from installed wheels."""
from __future__ import annotations

import importlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys


BASE_MODULES = {
    'rles_contracts': 'rles-contracts',
    'rles_data': 'rles-data',
    'rles_nautilus_adapter': 'rles-nautilus-adapter',
    'rles_execution': 'rles-execution',
    'rles_risk': 'rles-risk',
    'rles_gym_bridge': 'rles-gym-bridge',
    'rles_rl_sb3': 'rles-rl-sb3',
    'rles_es_evotorch': 'rles-es-evotorch',
    'rles_orchestrator': 'rles-orchestrator',
    'rles_compute_runtime': 'rles-compute-runtime',
    'rles_research': 'rles-research',
    'rles_il_core': 'rles-il-core',
    'rles_route_a': 'rles-route-a-imitation',
}


def discover_modules() -> dict[str, str]:
    packages = Path(__file__).resolve().parents[2] / 'packages'
    result = dict(BASE_MODULES)
    for path in packages.glob('*/pyproject.toml'):
        try:
            import tomllib
            metadata = tomllib.loads(path.read_text(encoding='utf-8'))
            distribution_name = metadata['project']['name']
            module_suffix = distribution_name.removeprefix('rles-').replace('-', '_')
        except (KeyError, OSError, ValueError):
            continue
        if module_suffix == 'route_a_imitation':
            module_suffix = 'route_a'
        module_name = 'rles_' + module_suffix
        previous = result.get(module_name)
        if previous is not None and previous != distribution_name:
            raise SystemExit(
                f'{path}: {module_name} maps to both {previous} and {distribution_name}'
            )
        result[module_name] = distribution_name
    return dict(sorted(result.items()))


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    records: dict[str, dict[str, str]] = {}
    for name, distribution_name in discover_modules().items():
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise SystemExit(f'{name}: missing installed package path')
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise SystemExit(f'{name}: imported outside wheel environment: {path}')
        dist = distribution(distribution_name)
        direct_url = dist.read_text('direct_url.json')
        if direct_url is None:
            raise SystemExit(f'{name}: local wheel provenance missing')
        origin = json.loads(direct_url)
        if (origin.get('dir_info', {}).get('editable')
            or not isinstance(origin.get('archive_info'), dict)
            or not origin.get('url', '').startswith('file:')
            or not origin.get('url', '').lower().endswith('.whl')):
            raise SystemExit(f'{name}: expected archive installation, got {origin}')
        records[name] = {'version': dist.version, 'module_path': str(path)}
    print(json.dumps({'python': sys.version.split()[0], 'prefix': str(prefix),
                      'installed_wheels': records}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
