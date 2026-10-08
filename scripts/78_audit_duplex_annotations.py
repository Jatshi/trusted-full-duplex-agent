#!/usr/bin/env python3
"""Reject unusable 3.0 annotations before loading the 9B feature extractor."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bootstrap import ROOT
from tfd.duplex_policy.data import grouped_entity_split


def audit(rows: list[dict], *, seed: int = 42, min_dialogues: int = 40) -> dict:
    if not rows:
        raise ValueError("annotation manifest is empty")
    ids = [str(row["dialogue_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dialogue_id must be unique")
    problems: list[str] = []
    if len(rows) < min_dialogues:
        problems.append(f"only {len(rows)} dialogues; require at least {min_dialogues}")
    pairs = [(row["dialogue_id"], str(row.get("split_group") or row["dialogue_id"])) for row in rows]
    try:
        splits = grouped_entity_split(pairs, seed=seed)
    except ValueError as exc:
        splits = {name: [] for name in ("train", "val", "calibration", "test")}
        problems.append(f"cannot make leakage-safe four-way split: {exc}")
    by_id = {row["dialogue_id"]: row for row in rows}
    counts: dict[str, dict] = {}

    audio_groups = defaultdict(set)
    speaker_groups = defaultdict(set)
    conversation_groups = defaultdict(set)
    hash_groups = defaultdict(set)
    for row in rows:
        condition = row.get("condition", {})
        if condition.get("scientific_label") is not True or condition.get("weak_label") is True:
            problems.append(f"{row['dialogue_id']}: lacks a reviewed scientific label")
        if not str(row.get("source", "")).strip() or not str(row.get("license", "")).strip():
            problems.append(f"{row['dialogue_id']}: missing source or license")
        if not str(row.get("consent_id", "")).strip():
            problems.append(f"{row['dialogue_id']}: missing consent record ID")
        for field in ("speaker_id", "conversation_id", "sha256"):
            if not str(row.get(field, "")).strip():
                problems.append(f"{row['dialogue_id']}: missing {field}")
        if row.get("annotation_complete") is not True:
            problems.append(f"{row['dialogue_id']}: annotation not marked complete")
        if row.get("action_intervals") is None and row.get("actions") is None:
            problems.append(f"{row['dialogue_id']}: missing action judgement")
        if row.get("risk_intervals") is None and row.get("risk_labels") is None:
            problems.append(f"{row['dialogue_id']}: missing risk judgement")
        if not str(row.get("split_group", "")).strip():
            problems.append(f"{row['dialogue_id']}: missing explicit split_group")
        if row.get("user_system_tracks_verified") is not True:
            problems.append(f"{row['dialogue_id']}: user/system tracks not verified")
        if not str(row.get("system_audio_path") or "").strip():
            problems.append(f"{row['dialogue_id']}: missing separate system audio path")
        if not str(row.get("system_sha256") or "").strip():
            problems.append(f"{row['dialogue_id']}: missing system audio hash")
        if row.get("system_active_intervals") is None:
            problems.append(f"{row['dialogue_id']}: missing playback activity judgement")
        group = str(row.get("split_group") or row["dialogue_id"])
        audio_groups[str(row.get("audio_path", ""))].add(group)
        for field, mapping in (("speaker_id", speaker_groups),
                               ("conversation_id", conversation_groups),
                               ("sha256", hash_groups)):
            key = str(row.get(field) or "").strip()
            if key:
                mapping[key].add(group)
    for audio, groups in audio_groups.items():
        if audio and len(groups) > 1:
            problems.append(f"audio {audio!r} occurs in multiple split groups")
    for name, mapping in (("speaker", speaker_groups), ("conversation", conversation_groups),
                          ("audio hash", hash_groups)):
        for entity, groups in mapping.items():
            if len(groups) > 1:
                problems.append(f"{name} {entity!r} occurs in multiple split groups")

    for split, members in splits.items():
        subset = [by_id[dialogue_id] for dialogue_id in members]
        events = {
            "take": sum(row.get("take_event_index", row.get("take_event_ms")) is not None for row in subset),
            "yield": sum(row.get("yield_event_index", row.get("yield_event_ms")) is not None for row in subset),
        }
        actions = Counter(
            action
            for row in subset
            for action in row.get("actions", [])
        )
        if not actions:
            actions = Counter(
                interval.get("action")
                for row in subset
                for interval in row.get("action_intervals", [])
            )
        counts[split] = {
            "dialogues": len(subset),
            "independent_groups": len({str(row.get("split_group") or row["dialogue_id"]) for row in subset}),
            "take_events": events["take"],
            "yield_events": events["yield"],
            "action_counts": dict(sorted(actions.items())),
            "scenarios": dict(sorted(Counter(
                row.get("condition", {}).get("scenario", "unspecified")
                for row in subset
            ).items())),
        }
        if not subset:
            problems.append(f"{split}: empty split")
        if split in ("val", "calibration", "test"):
            for row in subset:
                reviewers = row.get("reviewed_by") or []
                if len(set(reviewers)) < 2:
                    problems.append(f"{row['dialogue_id']}: {split} requires two independent reviewers")
        if split == "train":
            for event, count in events.items():
                if count < 3:
                    problems.append(f"train: only {count} {event.upper()} events; require at least 3")
        elif split == "test":
            for event, count in events.items():
                if count < 1:
                    problems.append(f"test: no {event.upper()} event to measure")
    if counts["calibration"]["independent_groups"] < 5:
        problems.append("calibration: fewer than 5 independent groups")

    return {
        "format": "tfd-star-annotation-readiness-v3",
        "seed": seed,
        "records": len(rows),
        "splits": counts,
        "ready_for_gpu_extraction": not problems,
        "problems": sorted(set(problems)),
        "claim_boundary": (
            "Passing this gate checks minimum annotation coverage only; it does not prove "
            "inter-annotator agreement or model quality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/duplex_policy_v3/annotation_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-dialogues", type=int, default=40)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = audit(rows, seed=args.seed, min_dialogues=args.min_dialogues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_gpu_extraction"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
