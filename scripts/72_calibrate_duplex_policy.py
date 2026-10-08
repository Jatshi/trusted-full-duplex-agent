#!/usr/bin/env python3
"""Fit held-out split-conformal action sets for the 3.0 policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bootstrap import load_config, resolve
from tfd.duplex_policy.calibration import SplitConformalClassifier
from tfd.duplex_policy.data import collate_duplex_sequences
from tfd.duplex_policy.feature_store import dialogue_ids, dialogue_split_groups, load_sequence_dataset, sha256_file, validate_split_manifest
from tfd.duplex_policy.metrics import brier_score, expected_calibration_error
from tfd.duplex_policy.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config("duplex_policy")
    out_dir = Path(resolve(cfg["paths"]["output_dir"]))
    checkpoint = args.checkpoint or out_dir / "best.pt"
    data_path = args.data or Path(resolve(cfg["paths"]["smoke_data"]))
    split_path = args.splits or Path(resolve(cfg["paths"]["split_manifest"]))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = set(split_payload["splits"]["calibration"])
    full = load_sequence_dataset(data_path)
    validate_split_manifest(full, split_payload, data_path)
    group_by_dialogue = dict(dialogue_split_groups(full))
    indices = [index for index, value in enumerate(dialogue_ids(full)) if value in allowed]
    if not indices:
        raise ValueError("calibration split is empty")
    dataset = Subset(full, indices)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_duplex_sequences)
    model, ckpt = load_checkpoint(checkpoint, map_location=args.device)
    device = torch.device(args.device)
    model.to(device).eval()
    all_probs, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = raw_batch.to(device)
            output = model(batch.acoustic, batch.semantic, batch.valid_mask)
            probs = output.action_logits.softmax(-1)[batch.valid_mask]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.actions[batch.valid_mask].cpu().numpy())
            for dialogue_id, valid in zip(batch.dialogue_ids, batch.valid_mask):
                all_groups.extend([group_by_dialogue[dialogue_id]] * int(valid.sum().item()))
    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    calibrator = SplitConformalClassifier(alpha=float(cfg["calibration"]["alpha"]))
    calibrator.fit(probabilities, labels, group_ids=all_groups)
    prediction_sets = calibrator.predict_sets(probabilities)
    coverage = float(np.mean([int(y in s) for y, s in zip(labels, prediction_sets)]))
    simultaneous_coverage = float(np.mean([
        all(labels[index] in prediction_sets[index]
            for index, observed_group in enumerate(all_groups) if observed_group == group)
        for group in set(all_groups)
    ]))
    report = {
        "format": "tfd-star-duplex-policy-calibration-v3",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "data_sha256": sha256_file(data_path),
        "splits_sha256": sha256_file(split_path),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "split": "calibration",
        "scientific_evidence_eligible": bool(
            split_payload.get("scientific_evidence_eligible", False)
        ),
        "weak_label_only": bool(split_payload.get("weak_label_only")),
        "num_frames": int(len(labels)),
        "num_dialogues": len(dataset),
        "num_independent_groups": len(set(all_groups)),
        "calibrator": calibrator.state_dict(),
        "metrics": {
            "empirical_coverage": coverage,
            "group_simultaneous_coverage": simultaneous_coverage,
            "mean_set_size": float(np.mean([len(value) for value in prediction_sets])),
            "ece": expected_calibration_error(
                probabilities, labels, bins=int(cfg["calibration"]["ece_bins"])
            ),
            "brier": brier_score(probabilities, labels),
        },
        "scope_note": (
            "group-max split conformal targets simultaneous frame coverage within "
            "an exchangeable speaker/conversation group; not a distribution-shift safety guarantee; "
            "weak/smoke calibration is pipeline validation only"
        ),
    }
    output = args.output or out_dir / "calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[ok] calibration -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
