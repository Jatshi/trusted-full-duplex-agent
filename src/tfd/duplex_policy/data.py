"""Validated JSONL sequence format and leakage-resistant batching/splitting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .actions import Action


@dataclass(frozen=True)
class DuplexSequence:
    dialogue_id: str
    frame_ms: int
    acoustic: torch.Tensor
    semantic: torch.Tensor
    actions: torch.Tensor
    risk_labels: torch.Tensor
    take_event_index: int
    yield_event_index: int
    source: str
    license: str
    condition: dict


@dataclass(frozen=True)
class DuplexBatch:
    dialogue_ids: list[str]
    acoustic: torch.Tensor
    semantic: torch.Tensor
    actions: torch.Tensor
    risk_labels: torch.Tensor
    valid_mask: torch.Tensor
    take_event_index: torch.Tensor
    yield_event_index: torch.Tensor

    def to(self, device: torch.device | str) -> "DuplexBatch":
        return DuplexBatch(
            dialogue_ids=self.dialogue_ids,
            acoustic=self.acoustic.to(device),
            semantic=self.semantic.to(device),
            actions=self.actions.to(device),
            risk_labels=self.risk_labels.to(device),
            valid_mask=self.valid_mask.to(device),
            take_event_index=self.take_event_index.to(device),
            yield_event_index=self.yield_event_index.to(device),
        )


class DuplexSequenceDataset(Dataset):
    def __init__(self, sequences: list[DuplexSequence]):
        if not sequences:
            raise ValueError("dataset must contain at least one sequence")
        acoustic_dim = sequences[0].acoustic.shape[1]
        semantic_dim = sequences[0].semantic.shape[1]
        for item in sequences:
            if item.acoustic.shape[1] != acoustic_dim:
                raise ValueError("acoustic feature dimension drift across records")
            if item.semantic.shape[1] != semantic_dim:
                raise ValueError("semantic feature dimension drift across records")
        self.sequences = sequences
        self.acoustic_dim = acoustic_dim
        self.semantic_dim = semantic_dim

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> DuplexSequence:
        return self.sequences[index]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DuplexSequenceDataset":
        source = Path(path)
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    rows.append(_parse_record(json.loads(raw), line_number))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
        return cls(rows)


def _parse_record(record: dict, line_number: int) -> DuplexSequence:
    if str(record.get("schema_version")) != "3.0":
        raise ValueError("schema_version must be '3.0'")
    dialogue_id = str(record["dialogue_id"]).strip()
    if not dialogue_id:
        raise ValueError("dialogue_id cannot be empty")
    frame_ms = int(record["frame_ms"])
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    acoustic = _finite_matrix(record["acoustic"], "acoustic")
    semantic = _finite_matrix(record["semantic"], "semantic")
    if len(acoustic) != len(semantic):
        raise ValueError("acoustic and semantic streams must have equal length")
    action_values = [Action.from_label(value).value for value in record["actions"]]
    risk_values = [int(value) for value in record["risk_labels"]]
    length = len(acoustic)
    if len(action_values) != length or len(risk_values) != length:
        raise ValueError("actions and risk_labels must align with every frame")
    if any(value < 0 or value > 2 for value in risk_values):
        raise ValueError("risk labels must be 0, 1 or 2")
    take = _event_index(record.get("take_event_index"), length, "take_event_index")
    yield_event = _event_index(record.get("yield_event_index"), length, "yield_event_index")
    return DuplexSequence(
        dialogue_id=dialogue_id,
        frame_ms=frame_ms,
        acoustic=torch.tensor(acoustic, dtype=torch.float32),
        semantic=torch.tensor(semantic, dtype=torch.float32),
        actions=torch.tensor(action_values, dtype=torch.long),
        risk_labels=torch.tensor(risk_values, dtype=torch.long),
        take_event_index=take,
        yield_event_index=yield_event,
        source=str(record["source"]),
        license=str(record["license"]),
        condition=dict(record.get("condition", {})),
    )


def _finite_matrix(values: list, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} feature dimension is inconsistent") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _event_index(value: int | None, length: int, name: str) -> int:
    if value is None:
        return -1
    index = int(value)
    if not 0 <= index < length:
        raise ValueError(f"{name} is outside sequence length")
    return index


def collate_duplex_sequences(items: list[DuplexSequence]) -> DuplexBatch:
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([len(item.actions) for item in items], dtype=torch.long)
    max_len = int(lengths.max().item())
    valid = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return DuplexBatch(
        dialogue_ids=[item.dialogue_id for item in items],
        acoustic=pad_sequence([item.acoustic for item in items], batch_first=True),
        semantic=pad_sequence([item.semantic for item in items], batch_first=True),
        actions=pad_sequence([item.actions for item in items], batch_first=True),
        risk_labels=pad_sequence([item.risk_labels for item in items], batch_first=True),
        valid_mask=valid,
        take_event_index=torch.tensor([item.take_event_index for item in items]),
        yield_event_index=torch.tensor([item.yield_event_index for item in items]),
    )


def grouped_dialogue_split(
    dialogue_ids: Iterable[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
) -> dict[str, list[str]]:
    """Deterministic group split; a dialogue can never cross split boundaries."""
    names = ("train", "val", "calibration", "test")
    if len(ratios) != 4 or any(value <= 0 for value in ratios):
        raise ValueError("four positive split ratios are required")
    total = sum(ratios)
    cumulative = np.cumsum(np.asarray(ratios, dtype=np.float64) / total)
    unique = sorted(set(str(value) for value in dialogue_ids))
    if len(unique) < 4:
        raise ValueError("at least four unique dialogues are needed for four-way splitting")
    result = {name: [] for name in names}
    for dialogue_id in unique:
        digest = hashlib.sha256(f"{seed}:{dialogue_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        split_index = int(np.searchsorted(cumulative, value, side="right"))
        result[names[min(split_index, 3)]].append(dialogue_id)
    # With very small corpora a hash bucket can be empty. Move one deterministic
    # item from the largest bucket so smoke tests and calibration remain usable.
    for empty in [name for name in names if not result[name]]:
        donor = max(names, key=lambda name: len(result[name]))
        if len(result[donor]) <= 1:
            raise ValueError("not enough dialogues to populate every split")
        result[empty].append(result[donor].pop())
    return result


def grouped_entity_split(
    dialogue_to_group: Iterable[tuple[str, str]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
) -> dict[str, list[str]]:
    """Split dialogue IDs while keeping every speaker/source group together."""
    pairs = [(str(dialogue), str(group)) for dialogue, group in dialogue_to_group]
    if not pairs or len({dialogue for dialogue, _ in pairs}) != len(pairs):
        raise ValueError("dialogue IDs must be non-empty and unique")
    groups = sorted({group for _, group in pairs})
    group_splits = grouped_dialogue_split(groups, seed=seed, ratios=ratios)
    group_to_split = {
        group: split_name
        for split_name, members in group_splits.items()
        for group in members
    }
    result = {name: [] for name in group_splits}
    for dialogue, group in pairs:
        result[group_to_split[group]].append(dialogue)
    return result
