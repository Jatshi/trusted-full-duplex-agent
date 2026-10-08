"""Create provenance manifest from all externally provisioned trained arms."""
import argparse
import json
from pathlib import Path
from lab import sha

ROOT = Path(__file__).resolve().parent


def prepare(parent, prefix):
    files = [parent/'encode/features.pt', parent/'data.json']
    files.extend(ROOT/name for name in ('controlled.py','memory.py','lab.py','service.py','panel.html'))
    models = {}
    for seed in (42,43,44):
        for arm in ('memory','gru','no_clear'):
            path = parent/f'{arm}_{seed}/best.pt'
            models[f'E136_{arm}_{seed}'] = {'arm':arm,'path':str(path)}
            files.append(path)
        for arm in ('memory_endpoint','memory_prefix','gru_prefix'):
            path = prefix/f'{arm}_{seed}/best.pt'
            models[f'E137_{arm}_{seed}'] = {'arm':'gru' if arm=='gru_prefix' else 'memory','path':str(path)}
            files.append(path)
    return {'parent':str(parent),'checkpoints':models,'files':{str(path):sha(path) for path in files}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    pins = prepare(args.parent.resolve(strict=True), args.prefix.resolve(strict=True))
    with (ROOT/'assets.json').open('x', encoding='utf-8') as stream:
        json.dump(pins, stream, indent=2)
