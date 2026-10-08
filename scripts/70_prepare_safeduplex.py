#!/usr/bin/env python3
"""Validate SafeDuplex 3.0 JSONL and create leakage-resistant split manifests.

The built-in generator is explicitly smoke-only.  It proves plumbing and must
never be used to support a research performance claim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bootstrap import ROOT, load_config, resolve
from tfd.duplex_policy.data import grouped_entity_split
from tfd.duplex_policy.feature_store import dialogue_split_groups, load_sequence_dataset, sha256_file
from tfd.duplex_policy.synthetic import generate_synthetic_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config("duplex_policy")
    seed = args.seed if args.seed is not None else int(cfg["training"]["seed"])
    output = args.output_jsonl or Path(resolve(cfg["paths"]["smoke_data"]))
    if args.synthetic_smoke:
        rows = generate_synthetic_records(
            args.count,
            seed=seed,
            acoustic_dim=int(cfg["features"]["acoustic_dim"]),
            semantic_dim=int(cfg["features"]["semantic_dim"]),
            frame_ms=int(cfg["clock"]["frame_ms"]),
        )
        source = write_jsonl(rows, output)
    else:
        source = args.input_jsonl.resolve()

    dataset = load_sequence_dataset(source)
    splits = grouped_entity_split(dialogue_split_groups(dataset), seed=seed)
    conditions = [dataset[index].condition for index in range(len(dataset))]
    scientific_evidence_eligible = all(
        condition.get("scientific_label") is True for condition in conditions
    )
    weak_label_only = any(condition.get("weak_label") is True for condition in conditions)
    split_path = args.split_manifest or Path(resolve(cfg["paths"]["split_manifest"]))
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "3.0",
        "seed": seed,
        "source_jsonl": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "synthetic_smoke_only": bool(args.synthetic_smoke),
        "weak_label_only": weak_label_only,
        "scientific_evidence_eligible": scientific_evidence_eligible,
        "acoustic_dim": dataset.acoustic_dim,
        "semantic_dim": dataset.semantic_dim,
        "records": len(dataset),
        "splits": splits,
    }
    split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scenarios = Counter(condition.get("scenario", "unspecified") for condition in conditions)
    print(f"[ok] validated {len(dataset)} sequences: {source}")
    print(f"     acoustic_dim={dataset.acoustic_dim} semantic_dim={dataset.semantic_dim}")
    print(f"     split sizes={ {name: len(ids) for name, ids in splits.items()} }")
    print(f"     manifest={split_path}")
    print(
        f"     scientific_evidence_eligible={scientific_evidence_eligible} "
        f"weak_label_only={weak_label_only}"
    )
    if args.synthetic_smoke:
        print("[warning] synthetic smoke data validates code only; it is not scientific evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
