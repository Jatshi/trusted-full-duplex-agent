#!/usr/bin/env python3
"""Convert millisecond event annotations to the frame-exact extraction schema."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from bootstrap import ROOT
from tfd.duplex_policy.actions import Action


def _frame(ms: float, frame_ms: int, frames: int) -> int:
    return min(frames - 1, max(0, int(math.floor(float(ms) / frame_ms))))


def _interval_bounds(interval: dict, duration_ms: int, frame_ms: int, frames: int) -> tuple[int, int]:
    start_ms = float(interval["start_ms"])
    end_ms = float(interval["end_ms"])
    if not math.isfinite(start_ms) or not math.isfinite(end_ms):
        raise ValueError("interval endpoints must be finite")
    if not 0 <= start_ms < end_ms <= duration_ms:
        raise ValueError("interval must satisfy 0 <= start_ms < end_ms <= duration_ms")
    return _frame(start_ms, frame_ms, frames), _frame(end_ms - 1e-6, frame_ms, frames)


def _event_frame(value: float | None, duration_ms: int, frame_ms: int, frames: int) -> int | None:
    if value is None:
        return None
    timestamp = float(value)
    if not math.isfinite(timestamp) or not 0 <= timestamp < duration_ms:
        raise ValueError("event time must be finite and within [0, duration_ms)")
    return _frame(timestamp, frame_ms, frames)


def normalize(record: dict, frame_ms: int) -> dict:
    if record.get("annotation_complete") is False:
        raise ValueError("annotation packet is not complete")
    if record.get("action_intervals") is None or record.get("risk_intervals") is None:
        raise ValueError("action and risk intervals must be explicitly judged")
    duration_ms = int(record["duration_ms"])
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    frames = int(math.ceil(duration_ms / frame_ms))
    system_intervals = record.get("system_active_intervals")
    if system_intervals is not None:
        for interval in system_intervals:
            _interval_bounds(interval, duration_ms, frame_ms, frames)
    actions = [Action.LISTEN.label] * frames
    risks = [0] * frames
    occupied = [False] * frames
    for interval in record.get("action_intervals", []):
        action = Action.from_label(interval["action"]).label
        start, end = _interval_bounds(interval, duration_ms, frame_ms, frames)
        for index in range(start, end + 1):
            if occupied[index] and actions[index] != action:
                raise ValueError(f"conflicting action intervals at frame {index}")
            actions[index] = action
            occupied[index] = True
    for interval in record.get("risk_intervals", []):
        level = int(interval["level"])
        if level not in (0, 1, 2):
            raise ValueError("risk level must be 0, 1 or 2")
        start, end = _interval_bounds(interval, duration_ms, frame_ms, frames)
        for index in range(start, end + 1):
            risks[index] = max(risks[index], level)
    take = record.get("take_event_ms")
    yield_event = record.get("yield_event_ms")
    return {
        "schema_version": "3.0-extraction",
        "dialogue_id": str(record["dialogue_id"]),
        # JSONL paths are portable POSIX-relative paths, even when prepared
        # on Windows and consumed by the Linux AutoDL extractor.
        "audio_path": str(record["audio_path"]).replace("\\", "/"),
        "system_audio_path": record.get("system_audio_path"),
        "system_sha256": record.get("system_sha256"),
        "system_active_intervals": system_intervals,
        "duration_ms": duration_ms,
        "frame_ms": frame_ms,
        "actions": actions,
        "risk_labels": risks,
        "take_event_index": _event_frame(take, duration_ms, frame_ms, frames),
        "yield_event_index": _event_frame(yield_event, duration_ms, frame_ms, frames),
        "source": str(record["source"]),
        "license": str(record["license"]),
        "consent_id": record.get("consent_id"),
        "sha256": record.get("sha256"),
        "annotation_complete": record.get("annotation_complete"),
        "reviewed_by": record.get("reviewed_by"),
        "user_system_tracks_verified": record.get("user_system_tracks_verified"),
        "condition": dict(record.get("condition", {})),
        "speaker_id": record.get("speaker_id"),
        "conversation_id": record.get("conversation_id", record["dialogue_id"]),
        "split_group": str(
            record.get("split_group")
            or record.get("speaker_id")
            or record.get("conversation_id")
            or record["dialogue_id"]
        ),
    }


def relocate_audio_path(row: dict, *, input_dir: Path, output_dir: Path) -> dict:
    """Keep a raw-manifest-relative recording valid after writing elsewhere."""
    relocated = dict(row)
    for field in ("audio_path", "system_audio_path"):
        value = row.get(field)
        if not value:
            continue
        audio = Path(value)
        resolved = (audio if audio.is_absolute() else input_dir / audio).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        relocated[field] = Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()
    return relocated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/safeduplex_cn/interim/extraction.jsonl")
    parser.add_argument("--frame-ms", type=int, default=160)
    args = parser.parse_args()
    if args.frame_ms <= 0:
        raise ValueError("frame-ms must be positive")
    rows = []
    for line_number, raw in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = relocate_audio_path(json.loads(raw), input_dir=args.input.resolve().parent,
                                         output_dir=args.output.resolve().parent)
            rows.append(normalize(record, args.frame_ms))
        except Exception as exc:
            raise ValueError(f"{args.input}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("input annotation file is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"[ok] normalized {len(rows)} dialogues -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
