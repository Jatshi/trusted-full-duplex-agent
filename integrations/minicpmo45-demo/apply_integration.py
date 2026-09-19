#!/usr/bin/env python3
"""Apply the TFD-STAR 2.0 delta to a MiniCPM-o-Demo checkout.

The upstream repository did not expose a repository-level license at release
time, so this package distributes only a source diff plus files authored by this
project. It does not vendor the upstream tree.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH = ROOT / "tfd-star-v2.patch"
OVERLAY = ROOT / "overlay"


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_root", type=Path, help="MiniCPM-o-Demo checkout")
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()
    demo = args.demo_root.expanduser().resolve()
    if not (demo / ".git").is_dir():
        parser.error(f"not a Git checkout: {demo}")

    run("git", "apply", "--check", str(PATCH), cwd=demo)
    if args.check:
        print("patch applies cleanly")
        return 0

    run("git", "apply", str(PATCH), cwd=demo)
    copied = 0
    for source in OVERLAY.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(OVERLAY)
        target = demo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"TFD-STAR patch applied; copied {copied} project-owned files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
