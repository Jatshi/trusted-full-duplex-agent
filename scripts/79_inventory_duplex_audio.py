#!/usr/bin/env python3
"""Inventory candidate WAV recordings without inventing consent or labels.

The CSV is an annotation queue, not a scientific dataset. In particular,
stereo channels are not assumed to be separated user/system tracks, and a
content hash is not a sufficient leakage group for augmented recordings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import wave
from collections import Counter
from pathlib import Path

import numpy as np


FIELDS = (
    "dialogue_id", "audio_path", "duration_ms", "sample_rate", "channels",
    "sample_width_bytes", "sha256", "rms_dbfs", "clipped_fraction",
    "conversation_id", "speaker_id", "split_group", "consent_id", "source",
    "license", "scenario", "user_system_tracks_verified", "annotation_status",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signal_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        if handle.getcomptype() != "NONE" or width not in (2, 4):
            raise ValueError("only uncompressed PCM16/PCM32 WAV is supported")
        if channels <= 0 or sample_rate <= 0 or frames <= 0:
            raise ValueError("recording has invalid channels, rate, or duration")
        dtype = np.dtype("<i2" if width == 2 else "<i4")
        peak_integer = (1 << (8 * width - 1)) - 1
        scale = float(1 << (8 * width - 1))
        sum_squares = 0.0
        clipped = 0
        samples_seen = 0
        while True:
            data = handle.readframes(65536)
            if not data:
                break
            samples = np.frombuffer(data, dtype=dtype)
            if not len(samples):
                break
            values = samples.astype(np.float64) / scale
            # Avoid BLAS startup/oversubscription for small audio chunks.
            sum_squares += float(np.square(values).sum(dtype=np.float64))
            clipped += int(np.count_nonzero(np.abs(samples.astype(np.int64)) >= peak_integer))
            samples_seen += len(samples)
    rms = math.sqrt(sum_squares / samples_seen) if samples_seen else 0.0
    return {
        "duration_ms": round(1000 * frames / sample_rate),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": width,
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2),
        "clipped_fraction": round(clipped / samples_seen, 6),
    }


def inventory(source_dir: Path, output_csv: Path) -> tuple[list[dict], list[str]]:
    files = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not files:
        raise ValueError(f"no WAV files found under {source_dir}")
    rows = []
    errors = []
    for path in files:
        try:
            stats = _signal_stats(path)
            digest = _hash_file(path)
            stem = re.sub(r"[^a-z0-9_-]+", "-", path.stem.lower()).strip("-") or "audio"
            row = {
                "dialogue_id": f"{stem[:50]}-{digest[:8]}",
                "audio_path": Path(os.path.relpath(path, output_csv.parent)).as_posix(),
                **stats,
                "sha256": digest,
                "conversation_id": "",
                "speaker_id": "",
                "split_group": "",
                "consent_id": "",
                "source": "",
                "license": "",
                "scenario": "",
                "user_system_tracks_verified": "",
                "annotation_status": "unreviewed",
            }
            rows.append(row)
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            errors.append(f"{path}: {exc}")
    return rows, errors


def quality_report(rows: list[dict], errors: list[str], source_dir: Path) -> str:
    rates = Counter(row["sample_rate"] for row in rows)
    channels = Counter(row["channels"] for row in rows)
    hashes = Counter(row["sha256"] for row in rows)
    unique = len(hashes)
    duration_sec = sum(row["duration_ms"] for row in rows) / 1000
    clipping = [row for row in rows if row["clipped_fraction"] > 0.001]
    too_short = [row for row in rows if row["duration_ms"] < 1000]
    valid_16k = sum(row["sample_rate"] == 16000 for row in rows)
    lines = [
        "# SafeDuplex-CN audio quality report",
        "",
        f"Source directory: `{source_dir}`",
        "",
        "## Four questions",
        "",
        f"- Enough data? No: {unique} unique files, {duration_sec:.1f} total seconds; pilot gate requires at least 40 independent reviewed dialogues.",
        f"- Reliable quality? {len(rows)} readable PCM WAV; {len(errors)} unreadable; {valid_16k}/{len(rows)} at 16 kHz; {len(clipping)} have >0.1% clipping.",
        "- Enough variety? Cannot establish it from files alone: speaker, conversation, scenario, consent, and separated-track metadata require review.",
        "- Useful features? These files can test acoustic feature extraction; they do not by themselves provide trustworthy TAKE/YIELD timing labels.",
        "",
        "## Counts",
        "",
        f"- Files scanned successfully: {len(rows)}",
        f"- Unique audio hashes: {unique}",
        f"- Duplicate audio copies: {len(rows) - unique}",
        f"- Sample rates: {dict(sorted(rates.items()))}",
        f"- Channel counts: {dict(sorted(channels.items()))}",
        f"- Recordings shorter than 1 second: {len(too_short)}",
        f"- Read/format errors: {len(errors)}",
        "",
        "## Interpretation",
        "",
        "This is an inventory. No consent, scientific label, user/system channel assignment, or augmentation-family grouping has been inferred. Do not feed this CSV directly to the GPU extractor.",
    ]
    if errors:
        lines.extend(["", "## File errors", "", *[f"- {error}" for error in errors]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    output = args.output_csv.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, errors = inventory(source, output)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = args.quality_report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(quality_report(rows, errors, source), encoding="utf-8")
    print(f"[inventory] {len(rows)} readable WAV, {len(errors)} errors -> {output}")
    print(f"[quality] {report}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
