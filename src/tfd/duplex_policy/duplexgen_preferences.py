"""Causal parsing of DuplexGen's human slot-level preferences.

These are English *assistant-listening* preferences, not observed audio events,
not assistant-playback interruptions, and not seven-action safety labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTIONS = ("silent", "backchannel", "take_floor")


@dataclass(frozen=True)
class PreferenceSlot:
    dialogue_id: str
    scenario: str
    license: str
    role: str
    word_index: int
    prefix: str
    context: str
    counts: tuple[int, int, int]

    @property
    def total_votes(self) -> int:
        return sum(self.counts)


def load_preference_slots(root: str | Path, split: str) -> list[PreferenceSlot]:
    """Parse only already-spoken text at each boundary, grouped by dialogue.

    The current turn after ``word_index`` and all later turns are deliberately
    excluded.  A dialogue is the statistical unit for splits and intervals.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be an official DuplexGen train or test split")
    files = sorted(Path(root).glob(f"*/{split}.jsonl"))
    if not files:
        raise ValueError(f"no DuplexGen {split} annotation files under {root}")
    slots: list[PreferenceSlot] = []
    seen_dialogues: set[str] = set()
    for source in files:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            dialogue_id = str(record["example_id"])
            if dialogue_id in seen_dialogues:
                raise ValueError(f"duplicate dialogue id: {dialogue_id}")
            seen_dialogues.add(dialogue_id)
            scenario = str(record["scenario"])
            if scenario != source.parent.name:
                raise ValueError(f"scenario/path mismatch: {source}:{line_number}")
            history = record["history"]
            if not isinstance(history, list):
                raise ValueError("history must be a list")
            prior: list[str] = []
            for turn in history:
                role = str(turn["role"])
                if role not in {"user", "assistant"}:
                    raise ValueError(f"invalid dialogue role: {role}")
                content = str(turn["content"])
                words = content.split()
                for boundary in turn["boundaries"]:
                    index = boundary["word_index"]
                    if not isinstance(index, int) or not 0 <= index <= len(words):
                        raise ValueError("word_index outside spoken turn")
                    raw_counts = boundary["counts"]
                    if set(raw_counts).difference(ACTIONS):
                        raise ValueError("unknown human preference label")
                    counts = tuple(int(raw_counts.get(action, 0)) for action in ACTIONS)
                    if (any(value < 0 for value in counts)
                            or sum(counts) != boundary["total_count"]
                            or sum(counts) == 0):
                        raise ValueError("vote counts do not match total_count")
                    prefix = " ".join(words[:index])
                    context = "\n".join([*prior[-4:], f"{role}: {prefix}"])
                    slots.append(PreferenceSlot(
                        dialogue_id=dialogue_id,
                        scenario=scenario,
                        license=str(record["license"]),
                        role=role,
                        word_index=index,
                        prefix=prefix,
                        context=context,
                        counts=counts,
                    ))
                prior.append(f"{role}: {content}")
    return slots


def soft_cross_entropy(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Vote-weighted proper score; each individual human vote has equal weight."""
    votes = np.asarray(counts, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if (votes.ndim != 2 or probs.shape != votes.shape or votes.shape[1] != len(ACTIONS)
            or not np.isfinite(votes).all() or not np.isfinite(probs).all()
            or (votes < 0).any() or (votes.sum(axis=1) <= 0).any()
            or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("invalid vote counts or model probabilities")
    return float(-(votes * np.log(np.clip(probs, 1e-12, 1))).sum() / votes.sum())
