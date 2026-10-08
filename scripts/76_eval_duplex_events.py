#!/usr/bin/env python3
"""Evaluate take/yield time-to-event predictions and write an auditable report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap import ROOT
from tfd.duplex_policy.metrics import event_timing_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/duplex_policy_v3/event_report.json")
    parser.add_argument("--frame-ms", type=int, default=160)
    parser.add_argument("--tolerance-ms", type=float, default=320.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("prediction file is empty")

    def values(name: str) -> np.ndarray:
        return np.asarray([-1 if row.get(name) is None else int(row[name]) for row in rows])

    report = {
        "format": "tfd-star-duplex-event-eval-v3",
        "frame_ms": args.frame_ms,
        "tolerance_ms": args.tolerance_ms,
        "take": event_timing_metrics(
            values("predicted_take_event_index"),
            values("target_take_event_index"),
            frame_ms=args.frame_ms,
            tolerance_ms=args.tolerance_ms,
        ),
        "yield": event_timing_metrics(
            values("predicted_yield_event_index"),
            values("target_yield_event_index"),
            frame_ms=args.frame_ms,
            tolerance_ms=args.tolerance_ms,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
