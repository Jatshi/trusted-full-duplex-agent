"""Scalable NPZ feature store with a small auditable JSONL manifest."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import Action
from .data import DuplexSequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(dataset, payload: dict, data_path: str | Path) -> None:
    """Reject stale or overlapping split manifests before any fit/evaluation."""
    names = {"train", "val", "calibration", "test"}
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != names:
        raise ValueError("split manifest must contain train/val/calibration/test")
    ids = dialogue_ids(dataset)
    if len(ids) != len(set(ids)):
        raise ValueError("dataset dialogue_id values must be unique")
    members = [str(value) for name in sorted(names) for value in splits[name]]
    if len(members) != len(set(members)) or set(members) != set(ids):
        raise ValueError("split manifest is overlapping, stale, or incomplete")
    if any(not splits[name] for name in names):
        raise ValueError("every split must be nonempty")
    group_by_dialogue = dict(dialogue_split_groups(dataset))
    group_locations: dict[str, str] = {}
    for split_name, split_ids in splits.items():
        for dialogue_id in split_ids:
            group = group_by_dialogue[str(dialogue_id)]
            previous = group_locations.setdefault(group, split_name)
            if previous != split_name:
                raise ValueError(f"split_group {group!r} leaks across {previous} and {split_name}")
    expected_hash = payload.get("source_sha256")
    if expected_hash and expected_hash != sha256_file(data_path):
        raise ValueError("split manifest source SHA256 differs from the dataset")


def write_feature_record(
    *,
    feature_dir: str | Path,
    manifest_path: str | Path,
    dialogue_id: str,
    acoustic: np.ndarray,
    semantic: np.ndarray,
    actions: list[str],
    risk_labels: list[int],
    frame_ms: int,
    take_event_index: int | None,
    yield_event_index: int | None,
    source: str,
    license_name: str,
    condition: dict,
    split_group: str | None = None,
) -> Path:
    feature_root = Path(feature_dir)
    manifest = Path(manifest_path)
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in dialogue_id)
    if not safe_id:
        raise ValueError("dialogue_id must contain at least one safe filename character")
    destination = feature_root / f"{safe_id}.npz"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite feature record: {destination}")
    acoustic_array = np.asarray(acoustic, dtype=np.float32)
    semantic_array = np.asarray(semantic, dtype=np.float32)
    if acoustic_array.ndim != 2 or semantic_array.ndim != 2:
        raise ValueError("acoustic and semantic features must be rank-2")
    if acoustic_array.shape[0] != semantic_array.shape[0]:
        raise ValueError("feature streams must have equal frame counts")
    length = acoustic_array.shape[0]
    if len(actions) != length or len(risk_labels) != length:
        raise ValueError("labels must align to feature frames")
    if not np.isfinite(acoustic_array).all() or not np.isfinite(semantic_array).all():
        raise ValueError("features must be finite")
    np.savez_compressed(
        destination,
        acoustic=acoustic_array.astype(np.float16),
        semantic=semantic_array.astype(np.float16),
    )
    relative = os.path.relpath(destination, manifest.parent).replace("\\", "/")
    record = {
        "schema_version": "3.0-feature-store",
        "dialogue_id": dialogue_id,
        "feature_file": relative,
        "frame_ms": int(frame_ms),
        "actions": actions,
        "risk_labels": [int(value) for value in risk_labels],
        "take_event_index": take_event_index,
        "yield_event_index": yield_event_index,
        "source": source,
        "license": license_name,
        "condition": condition,
        "split_group": str(split_group or dialogue_id),
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


class FeatureManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if record.get("schema_version") != "3.0-feature-store":
                    raise ValueError(f"line {line_number}: unsupported feature-store schema")
                self.records.append(record)
        if not self.records:
            raise ValueError("feature manifest is empty")
        first = self._load_arrays(self.records[0])
        self.acoustic_dim = int(first[0].shape[1])
        self.semantic_dim = int(first[1].shape[1])
        self.dialogue_ids = [str(record["dialogue_id"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DuplexSequence:
        record = self.records[index]
        acoustic, semantic = self._load_arrays(record)
        length = acoustic.shape[0]
        actions = [Action.from_label(value).value for value in record["actions"]]
        risks = [int(value) for value in record["risk_labels"]]
        if semantic.shape[0] != length or len(actions) != length or len(risks) != length:
            raise ValueError(f"{record['dialogue_id']}: features and labels are misaligned")
        if any(value not in (0, 1, 2) for value in risks):
            raise ValueError(f"{record['dialogue_id']}: risk labels must be 0, 1 or 2")
        take_index = -1 if record.get("take_event_index") is None else int(record["take_event_index"])
        yield_index = -1 if record.get("yield_event_index") is None else int(record["yield_event_index"])
        if take_index >= length or yield_index >= length or take_index < -1 or yield_index < -1:
            raise ValueError(f"{record['dialogue_id']}: event index is outside feature frames")
        return DuplexSequence(
            dialogue_id=str(record["dialogue_id"]),
            frame_ms=int(record["frame_ms"]),
            acoustic=torch.from_numpy(acoustic),
            semantic=torch.from_numpy(semantic),
            actions=torch.tensor(actions, dtype=torch.long),
            risk_labels=torch.tensor(risks, dtype=torch.long),
            take_event_index=take_index,
            yield_event_index=yield_index,
            source=str(record["source"]),
            license=str(record["license"]),
            condition=dict(record.get("condition", {})),
        )

    def _load_arrays(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        feature_path = (self.manifest_path.parent / record["feature_file"]).resolve()
        with np.load(feature_path, allow_pickle=False) as payload:
            acoustic = np.asarray(payload["acoustic"], dtype=np.float32)
            semantic = np.asarray(payload["semantic"], dtype=np.float32)
        if acoustic.ndim != 2 or semantic.ndim != 2:
            raise ValueError(f"{record['dialogue_id']}: feature arrays must be rank-2")
        if acoustic.shape[1] != getattr(self, "acoustic_dim", acoustic.shape[1]):
            raise ValueError(f"{record['dialogue_id']}: acoustic feature dimension drift")
        if semantic.shape[1] != getattr(self, "semantic_dim", semantic.shape[1]):
            raise ValueError(f"{record['dialogue_id']}: semantic feature dimension drift")
        return acoustic, semantic


def load_sequence_dataset(path: str | Path):
    """Load either the compact smoke JSONL or the scalable NPZ manifest."""
    source = Path(path)
    first = next((line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()), None)
    if first is None:
        raise ValueError("dataset manifest is empty")
    schema = json.loads(first).get("schema_version")
    if schema == "3.0-feature-store":
        return FeatureManifestDataset(source)
    from .data import DuplexSequenceDataset

    return DuplexSequenceDataset.from_jsonl(source)


def dialogue_ids(dataset) -> list[str]:
    if hasattr(dataset, "dialogue_ids"):
        return list(dataset.dialogue_ids)
    return [item.dialogue_id for item in dataset.sequences]


def dialogue_split_groups(dataset) -> list[tuple[str, str]]:
    """Return (dialogue, leakage-group) pairs without loading all NPZ arrays."""
    if isinstance(dataset, FeatureManifestDataset):
        return [
            (str(record["dialogue_id"]), str(record.get("split_group", record["dialogue_id"])))
            for record in dataset.records
        ]
    return [
        (
            item.dialogue_id,
            str(item.condition.get("split_group", item.dialogue_id)),
        )
        for item in dataset.sequences
    ]
