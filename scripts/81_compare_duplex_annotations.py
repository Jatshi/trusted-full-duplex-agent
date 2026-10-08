#!/usr/bin/env python3
"""Compare two independent human annotation packets; never auto-adjudicate."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _normalizer():
    # The numbered CLI filename is not an importable Python identifier.
    path = ROOT / "scripts/74_normalize_duplex_annotations.py"
    spec = importlib.util.spec_from_file_location("tfd_duplex_normalize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize


def _read(path: Path) -> dict[str, dict]:
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        dialogue_id = str(row["dialogue_id"])
        if dialogue_id in records:
            raise ValueError(f"{path}:{line_number}: duplicate {dialogue_id}")
        records[dialogue_id] = row
    return records


def compare(left: dict[str, dict], right: dict[str, dict], *, frame_ms: int = 160,
            event_tolerance_ms: int = 320) -> list[dict]:
    normalize = _normalizer()
    result = []
    for dialogue_id in sorted(left.keys() | right.keys()):
        a, b = left.get(dialogue_id), right.get(dialogue_id)
        issues = []
        if a is None or b is None:
            issues.append("missing_from_A" if a is None else "missing_from_B")
        else:
            if a.get("sha256") != b.get("sha256") or a.get("duration_ms") != b.get("duration_ms"):
                issues.append("audio_identity_mismatch")
            if a.get("annotator_id") == b.get("annotator_id"):
                issues.append("annotator_identity_not_independent")
            if a.get("annotation_complete") is not True or b.get("annotation_complete") is not True:
                issues.append("incomplete_annotation")
            if a.get("action_intervals") is None or b.get("action_intervals") is None:
                issues.append("missing_action_judgement")
            if a.get("risk_intervals") is None or b.get("risk_intervals") is None:
                issues.append("missing_risk_judgement")
            if not issues:
                try:
                    na, nb = normalize(a, frame_ms), normalize(b, frame_ms)
                    if na["actions"] != nb["actions"]:
                        issues.append("action_frame_disagreement")
                    if na["risk_labels"] != nb["risk_labels"]:
                        issues.append("risk_frame_disagreement")
                    for event in ("take_event_ms", "yield_event_ms"):
                        av, bv = a.get(event), b.get(event)
                        if (av is None) != (bv is None):
                            issues.append(f"{event}_presence_disagreement")
                        elif av is not None and abs(float(av) - float(bv)) > event_tolerance_ms:
                            issues.append(f"{event}_timing_disagreement")
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(f"invalid_annotation: {exc}")
        result.append({"dialogue_id": dialogue_id, "needs_adjudication": bool(issues),
                       "issues": ";".join(issues)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--frame-ms", type=int, default=160)
    parser.add_argument("--event-tolerance-ms", type=int, default=320)
    args = parser.parse_args()
    if args.frame_ms <= 0 or args.event_tolerance_ms < 0:
        raise ValueError("frame-ms must be positive and event-tolerance-ms nonnegative")
    rows = compare(_read(args.annotator_a), _read(args.annotator_b),
                   frame_ms=args.frame_ms, event_tolerance_ms=args.event_tolerance_ms)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dialogue_id", "needs_adjudication", "issues"))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(row["needs_adjudication"] for row in rows)
    print(f"[agreement] {len(rows)} dialogues, {flagged} need adjudication -> {args.output_csv}")
    return 0 if not flagged else 2


if __name__ == "__main__":
    raise SystemExit(main())
