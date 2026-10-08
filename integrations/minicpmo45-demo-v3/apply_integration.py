"""Apply the verified 3.0 delta to the pinned clean upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent


def apply(demo, check=False):
    pins=json.loads((ROOT/'MANIFEST.json').read_text())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=demo,text=True).strip()
    if revision != pins['upstream_revision']:
        raise ValueError('Checkout the pinned upstream revision first')
    patch=ROOT/'tfd-star-v3.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != pins['patch_sha256']:
        raise ValueError('Integration patch SHA mismatch')
    for row in pins['overlay']:
        relative=PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['path']:
            raise ValueError('Invalid overlay path')
        src=ROOT/'overlay'/row['path']
        if src.stat().st_size != row['bytes'] or hashlib.sha256(src.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Overlay asset mismatch')
        if (demo/row['path']).exists():
            raise FileExistsError('Use a clean checkout; existing overlay is not overwritten')
    subprocess.run(['git','apply','--check',str(patch)],cwd=demo,check=True)
    if check:
        return
    subprocess.run(['git','apply',str(patch)],cwd=demo,check=True)
    for row in pins['overlay']:
        target=demo/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'overlay'/row['path'],target)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demo_root',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    apply(args.demo_root.resolve(),args.check)
    print('3.0 integration verified' if args.check else '3.0 integration installed')
