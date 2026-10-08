#!/usr/bin/env python3
"""CPU-only gate before spending AutoDL time on native 9B feature extraction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wave_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth(),
                handle.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("expected 16 kHz mono uncompressed PCM16 WAV")
        return round(handle.getnframes() * 1000 / handle.getframerate())


def _audit(rows: list[dict]) -> dict:
    script = ROOT / "scripts/78_audit_duplex_annotations.py"
    spec = importlib.util.spec_from_file_location("tfd_duplex_annotation_audit", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit(rows)


def check(rows: list[dict], *, manifest_dir: Path, mode: str) -> dict:
    if mode not in ("probe", "formal"):
        raise ValueError("mode must be probe or formal")
    problems = []
    if not rows:
        problems.append("empty extraction manifest")
    ids = [str(row.get("dialogue_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        problems.append("dialogue_id must be nonempty and unique")
    system_count = 0
    for row in rows:
        identity = str(row.get("dialogue_id", "<missing>"))
        if row.get("schema_version") != "3.0-extraction":
            problems.append(f"{identity}: wrong schema_version")
        try:
            duration = int(row["duration_ms"])
            frame_ms = int(row["frame_ms"])
            if duration <= 0 or frame_ms <= 0:
                raise ValueError("duration/frame must be positive")
            expected = math.ceil(duration / frame_ms)
            if len(row["actions"]) != expected or len(row["risk_labels"]) != expected:
                problems.append(f"{identity}: labels do not match frame clock")
            for name in ("take_event_index", "yield_event_index"):
                value = row.get(name)
                if value is not None and not 0 <= int(value) < expected:
                    problems.append(f"{identity}: {name} outside frame clock")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{identity}: invalid frame/label schema: {exc}")
            continue
        paths = {}
        for field, digest_field in (("audio_path", "sha256"),
                                    ("system_audio_path", "system_sha256")):
            value = row.get(field)
            if not value:
                if field == "audio_path":
                    problems.append(f"{identity}: missing user audio path")
                if field == "system_audio_path" and mode == "formal":
                    problems.append(f"{identity}: missing system track")
                continue
            path = Path(value)
            path = (path if path.is_absolute() else manifest_dir / path).resolve()
            paths[field] = path
            try:
                actual_ms = _wave_ms(path)
                if abs(actual_ms - duration) > 40:
                    problems.append(f"{identity}: {field} duration mismatch ({actual_ms} vs {duration} ms)")
                declared = row.get(digest_field)
                if declared and str(declared).lower() != _hash(path):
                    problems.append(f"{identity}: {field} SHA256 mismatch")
                if mode == "formal" and not declared:
                    problems.append(f"{identity}: missing {digest_field}")
            except (OSError, EOFError, ValueError, wave.Error) as exc:
                problems.append(f"{identity}: invalid {field}: {exc}")
        if "system_audio_path" in paths:
            system_count += 1
            if paths.get("audio_path") == paths["system_audio_path"]:
                problems.append(f"{identity}: user and system tracks are the same file")
            intervals = row.get("system_active_intervals")
            if intervals is None:
                problems.append(f"{identity}: missing system playback activity intervals")
            elif not isinstance(intervals, list):
                problems.append(f"{identity}: playback activity intervals must be a list")
            else:
                for interval in intervals:
                    try:
                        start, end = float(interval["start_ms"]), float(interval["end_ms"])
                        if not (math.isfinite(start) and math.isfinite(end)
                                and 0 <= start < end <= duration):
                            raise ValueError("outside recording")
                    except (KeyError, TypeError, ValueError) as exc:
                        problems.append(f"{identity}: invalid playback interval: {exc}")
        elif mode == "probe" and row.get("condition", {}).get("scientific_label") is True:
            problems.append(f"{identity}: user-only probe cannot claim scientific labels")
    annotation_audit = None
    if mode == "formal" and rows:
        annotation_audit = _audit(rows)
        if not annotation_audit["ready_for_gpu_extraction"]:
            problems.append("formal annotation audit failed; see annotation_audit.problems")
    return {
        "format": "tfd-star-gpu-preflight-v3",
        "mode": mode,
        "records": len(rows),
        "system_track_records": system_count,
        "expected_acoustic_dim": 4118,
        "expected_semantic_dim": 4096,
        "ready_for_gpu_probe": not problems and mode == "probe",
        "ready_for_formal_extraction": not problems and mode == "formal",
        "problems": problems,
        "annotation_audit": annotation_audit,
        "claim_boundary": (
            "CPU preflight failed; do not start GPU extraction."
            if problems else
            "Probe only; weak/user-only audio cannot support model-quality claims."
            if mode == "probe" else
            "CPU data checks pass only; native 9B causality, shape and latency still require GPU validation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "formal"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/duplex_policy_v3/gpu_preflight.json")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    report = check(rows, manifest_dir=args.manifest.resolve().parent, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "annotation_audit"},
                     ensure_ascii=False, indent=2))
    return 0 if not report["problems"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
