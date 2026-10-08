#!/usr/bin/env python3
"""Evaluate action accuracy, calibration and selective risk on held-out dialogues."""
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
from tfd.duplex_policy.metrics import (
    brier_score,
    event_timing_metrics,
    expected_calibration_error,
    risk_coverage_curve,
)
from tfd.duplex_policy.actions import Action
from tfd.duplex_policy.shield import RiskAwareActionShield
from tfd.duplex_policy.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
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
    calibration_path = args.calibration or out_dir / "calibration.json"
    data_path = args.data or Path(resolve(cfg["paths"]["smoke_data"]))
    split_path = args.splits or Path(resolve(cfg["paths"]["split_manifest"]))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = set(split_payload["splits"]["test"])
    full = load_sequence_dataset(data_path)
    validate_split_manifest(full, split_payload, data_path)
    group_by_dialogue = dict(dialogue_split_groups(full))
    indices = [index for index, value in enumerate(dialogue_ids(full)) if value in allowed]
    if not indices:
        raise ValueError("test split is empty")
    dataset = Subset(full, indices)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_duplex_sequences)
    model, ckpt = load_checkpoint(checkpoint, map_location=args.device)
    model.to(args.device).eval()
    cal_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    for field, path in (("checkpoint_sha256", checkpoint), ("data_sha256", data_path),
                        ("splits_sha256", split_path)):
        if cal_payload.get(field) != sha256_file(path):
            raise ValueError(f"calibration {field} does not match this evaluation input")
    calibrator = SplitConformalClassifier.from_state_dict(cal_payload["calibrator"])

    all_probs, all_labels, all_risk_probs, all_risk_labels, all_ood, all_groups = [], [], [], [], [], []
    predicted_take, target_take, predicted_yield, target_yield = [], [], [], []
    hazard_threshold = float(cfg["evaluation"]["hazard_threshold"])
    with torch.no_grad():
        for raw_batch in loader:
            batch = raw_batch.to(args.device)
            output = model(batch.acoustic, batch.semantic, batch.valid_mask)
            all_probs.append(output.action_logits.softmax(-1)[batch.valid_mask].cpu().numpy())
            all_labels.append(batch.actions[batch.valid_mask].cpu().numpy())
            risk_probs = output.risk_alpha / output.risk_alpha.sum(-1, keepdim=True)
            epistemic = output.risk_alpha.shape[-1] / output.risk_alpha.sum(-1)
            all_risk_probs.append(risk_probs[batch.valid_mask].cpu().numpy())
            all_risk_labels.append(batch.risk_labels[batch.valid_mask].cpu().numpy())
            all_ood.append(epistemic[batch.valid_mask].cpu().numpy())
            for dialogue_id, valid in zip(batch.dialogue_ids, batch.valid_mask):
                all_groups.extend([group_by_dialogue[dialogue_id]] * int(valid.sum().item()))
            for row in range(len(batch.dialogue_ids)):
                length = int(batch.valid_mask[row].sum().item())
                take_hits = torch.nonzero(
                    output.take_hazard_logits[row, :length].sigmoid() >= hazard_threshold
                ).flatten()
                yield_hits = torch.nonzero(
                    output.yield_hazard_logits[row, :length].sigmoid() >= hazard_threshold
                ).flatten()
                predicted_take.append(int(take_hits[0].item()) if len(take_hits) else -1)
                predicted_yield.append(int(yield_hits[0].item()) if len(yield_hits) else -1)
                target_take.append(int(batch.take_event_index[row].item()))
                target_yield.append(int(batch.yield_event_index[row].item()))
    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    risk_probabilities = np.concatenate(all_risk_probs)
    risk_labels = np.concatenate(all_risk_labels)
    ood_scores = np.concatenate(all_ood)
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    errors = (predicted != labels).astype(np.float64)
    sets = calibrator.predict_sets(probabilities)
    group_simultaneous_coverage = float(np.mean([
        all(labels[index] in sets[index]
            for index, observed_group in enumerate(all_groups) if observed_group == group)
        for group in set(all_groups)
    ]))
    shield = RiskAwareActionShield(
        high_risk=float(cfg["safety"]["high_risk"]),
        ood_threshold=float(cfg["safety"]["ood_threshold"]),
    )
    shielded = np.asarray(
        [
            shield.decide(
                {Action(value) for value in action_set},
                risk=float(risk_row[2]),
                ood_score=float(ood),
            ).value
            for action_set, risk_row, ood in zip(sets, risk_probabilities, ood_scores)
        ],
        dtype=np.int64,
    )
    high_risk_frames = risk_labels == 2
    unsafe_before = (predicted == Action.EXECUTE.value) & high_risk_frames
    unsafe_after = (shielded == Action.EXECUTE.value) & high_risk_frames
    curve = risk_coverage_curve(confidence, errors)
    report = {
        "format": "tfd-star-duplex-policy-eval-v3",
        "checkpoint_epoch": int(ckpt["epoch"]),
        "split": "test",
        "synthetic_smoke_only": bool(split_payload.get("synthetic_smoke_only")),
        "weak_label_only": bool(split_payload.get("weak_label_only")),
        "scientific_evidence_eligible": bool(
            split_payload.get("scientific_evidence_eligible", False)
        ),
        "num_dialogues": len(dataset),
        "num_frames": int(len(labels)),
        "metrics": {
            "action_accuracy": float((predicted == labels).mean()),
            "shielded_action_accuracy": float((shielded == labels).mean()),
            "risk_accuracy": float((risk_probabilities.argmax(1) == risk_labels).mean()),
            "unsafe_execute_count_before_shield": int(unsafe_before.sum()),
            "unsafe_execute_count_after_shield": int(unsafe_after.sum()),
            "unsafe_execute_rate_before_shield": float(
                unsafe_before.sum() / max(1, high_risk_frames.sum())
            ),
            "unsafe_execute_rate_after_shield": float(
                unsafe_after.sum() / max(1, high_risk_frames.sum())
            ),
            "ece": expected_calibration_error(
                probabilities, labels, bins=int(cfg["calibration"]["ece_bins"])
            ),
            "brier": brier_score(probabilities, labels),
            "conformal_coverage": float(np.mean([int(y in s) for y, s in zip(labels, sets)])),
            "group_simultaneous_coverage": group_simultaneous_coverage,
            "mean_prediction_set_size": float(np.mean([len(value) for value in sets])),
            "selective_risk_at_50pct": min(curve, key=lambda item: abs(item[0] - 0.5))[1],
            "selective_risk_at_80pct": min(curve, key=lambda item: abs(item[0] - 0.8))[1],
        },
        "event_metrics": {
            "take": event_timing_metrics(
                np.asarray(predicted_take),
                np.asarray(target_take),
                frame_ms=int(cfg["clock"]["frame_ms"]),
                tolerance_ms=float(cfg["evaluation"]["event_tolerance_ms"]),
            ),
            "yield": event_timing_metrics(
                np.asarray(predicted_yield),
                np.asarray(target_yield),
                frame_ms=int(cfg["clock"]["frame_ms"]),
                tolerance_ms=float(cfg["evaluation"]["event_tolerance_ms"]),
            ),
        },
        "risk_coverage_curve": [{"coverage": c, "risk": r} for c, r in curve],
        "claim_boundary": (
            "This split is not eligible for scientific claims; it validates mechanics only."
            if not split_payload.get("scientific_evidence_eligible", False)
            else "Real-data result; multi-seed, public-benchmark and human evaluation remain required."
        ),
    }
    output = args.output or out_dir / "test_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[ok] test report -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
