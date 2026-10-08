"""Deterministic smoke data for plumbing tests, never for research claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .actions import Action


SCENARIOS = ("complete", "hesitation", "risky", "bargein")


def generate_synthetic_records(
    count: int,
    *,
    seed: int = 42,
    acoustic_dim: int = 8,
    semantic_dim: int = 12,
    frame_ms: int = 160,
) -> list[dict]:
    """Create causal toy sequences solely to validate the local pipeline."""
    if count <= 0 or acoustic_dim < 4 or semantic_dim < 4 or frame_ms <= 0:
        raise ValueError("invalid synthetic-data dimensions")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        length = int(rng.integers(10, 25))
        progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
        acoustic = rng.normal(0.0, 0.08, size=(length, acoustic_dim)).astype(np.float32)
        semantic = rng.normal(0.0, 0.05, size=(length, semantic_dim)).astype(np.float32)

        # First channels have documented meanings in the smoke generator.  Real
        # runs replace them with backbone/SSL states and preserve the schema.
        acoustic[:, 0] = 1.0                         # voiced probability
        acoustic[:, 1] = progress                    # elapsed normalized time
        acoustic[:, 2] = 0.0                         # overlap probability
        acoustic[:, 3] = 0.0                         # trailing silence
        semantic[:, 0] = progress                    # accumulated semantic evidence
        semantic[:, 1] = 0.0                         # risk belief
        semantic[:, 2] = 0.0                         # ambiguity belief
        semantic[:, 3] = 1.0 - progress              # epistemic uncertainty proxy

        actions = [Action.LISTEN.label] * length
        risks = [0] * length
        take_event = None
        yield_event = None
        if scenario == "complete":
            silence_start = max(length - 3, 1)
            acoustic[silence_start:, 0] = 0.0
            acoustic[silence_start:, 3] = np.linspace(0.3, 1.0, length - silence_start)
            actions[-1] = Action.TAKE.label
            take_event = length - 1
        elif scenario == "hesitation":
            middle = length // 2
            acoustic[middle : middle + 2, 0] = 0.0
            acoustic[middle : middle + 2, 3] = 0.35
            semantic[middle : middle + 2, 2] = 0.8
            actions[middle] = Action.BACKCHANNEL.label
            actions[-1] = Action.TAKE.label
            take_event = length - 1
        elif scenario == "risky":
            onset = length // 2
            semantic[onset:, 1] = np.linspace(0.4, 1.0, length - onset)
            risks[onset:] = [1] * max(length - onset - 2, 0) + [2] * min(2, length - onset)
            actions[-2:] = [Action.CLARIFY.label, Action.STOP.label]
        else:  # user barge-in while the agent is speaking
            onset = length // 2
            acoustic[onset:, 2] = 1.0
            actions[onset] = Action.YIELD.label
            yield_event = onset

        rows.append(
            {
                "schema_version": "3.0",
                "dialogue_id": f"synthetic-{seed}-{index:06d}",
                "frame_ms": frame_ms,
                "source": "tfd-star-synthetic-smoke-only",
                "license": "Apache-2.0",
                "scenario": scenario,
                "acoustic": acoustic.tolist(),
                "semantic": semantic.tolist(),
                "actions": actions,
                "risk_labels": risks,
                "take_event_index": take_event,
                "yield_event_index": yield_event,
                "condition": {
                    "scenario": scenario,
                    "snr_db": 20.0,
                    "overlap": scenario == "bargein",
                    "synthetic_smoke_only": True,
                },
            }
        )
    return rows


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return destination
