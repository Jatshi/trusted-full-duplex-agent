"""Pin externally provisioned checkpoints; never download, train or overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
from tfd.duplex_policy.demo_lora import LORA_PROFILES, verify_asset


def prepare(paths):
    if set(paths) != set(LORA_PROFILES):
        raise ValueError('All six named checkpoint directories required')
    entries = {}
    for profile, directory in paths.items():
        path = Path(directory).resolve(strict=True)
        entry = {'path': str(path), 'adapter_name': profile}
        for name, field in [('adapter_config.json', 'config_sha256'),
                            ('adapter_model.safetensors', 'weight_sha256')]:
            entry[field] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        verify_asset(entry)
        entries[profile] = entry
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    entries = prepare(json.loads(args.paths.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(entries, stream, indent=2)
