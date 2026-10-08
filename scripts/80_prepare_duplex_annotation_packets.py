#!/usr/bin/env python3
"""Create two *unfilled* independent annotation packets from an audio inventory.

Packets are deliberately invalid as model-training annotations until humans
complete the labels and provenance. Never infer consent or event timings.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def prepare(rows: list[dict], *, inventory_dir: Path, packet_dir: Path, annotator: str) -> list[dict]:
    packet = []
    seen = set()
    for row in rows:
        dialogue_id = str(row["dialogue_id"]).strip()
        if not dialogue_id or dialogue_id in seen:
            raise ValueError(f"missing or duplicate dialogue_id: {dialogue_id!r}")
        seen.add(dialogue_id)
        audio_path = (inventory_dir / row["audio_path"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        duration_ms = int(row["duration_ms"])
        if duration_ms <= 0:
            raise ValueError(f"{dialogue_id}: invalid duration")
        packet.append({
            "dialogue_id": dialogue_id,
            "audio_path": Path(os.path.relpath(audio_path, packet_dir)).as_posix(),
            "duration_ms": duration_ms,
            "sha256": row["sha256"],
            "system_audio_path": "",
            "system_sha256": "",
            "system_active_intervals": None,
            "annotator_id": annotator,
            "annotation_complete": False,
            "source": row.get("source", ""),
            "license": row.get("license", ""),
            "consent_id": row.get("consent_id", ""),
            "conversation_id": row.get("conversation_id", ""),
            "speaker_id": row.get("speaker_id", ""),
            "split_group": row.get("split_group", ""),
            "user_system_tracks_verified": row.get("user_system_tracks_verified", ""),
            "condition": {"scenario": row.get("scenario", ""), "scientific_label": False},
            # null means *not yet judged*. [] or null event after review may be
            # a legitimate negative/censored label, but must be chosen by a human.
            "action_intervals": None,
            "risk_intervals": None,
            "take_event_ms": None,
            "yield_event_ms": None,
            "annotation_notes": "",
        })
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = args.inventory_csv.resolve()
    output = args.output_dir.resolve()
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("inventory is empty")
    output.mkdir(parents=True, exist_ok=True)
    for annotator in ("A", "B"):
        packet = prepare(rows, inventory_dir=inventory.parent, packet_dir=output, annotator=annotator)
        path = output / f"annotator_{annotator}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet), encoding="utf-8")
        print(f"[packet] {len(packet)} unfilled records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
