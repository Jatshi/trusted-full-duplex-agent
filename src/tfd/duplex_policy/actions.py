"""Canonical action vocabulary shared by training, calibration and serving."""
from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    LISTEN = 0
    BACKCHANNEL = 1
    TAKE = 2
    YIELD = 3
    CLARIFY = 4
    STOP = 5
    EXECUTE = 6

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def from_label(cls, value: str) -> "Action":
        try:
            return cls[value.strip().upper()]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"unknown duplex action: {value!r}") from exc


ACTIONS = tuple(Action)
ACTION_LABELS = tuple(action.label for action in ACTIONS)
