"""Deterministic last-mile safety shield around a calibrated policy set."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .actions import Action


@dataclass(frozen=True)
class RiskAwareActionShield:
    high_risk: float = 0.7
    ood_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.high_risk <= 1.0:
            raise ValueError("high_risk must be in [0,1]")
        if not 0.0 <= self.ood_threshold <= 1.0:
            raise ValueError("ood_threshold must be in [0,1]")

    def decide(
        self,
        prediction_set: Iterable[Action | int],
        *,
        risk: float,
        ood_score: float,
    ) -> Action:
        actions = {Action(value) for value in prediction_set}
        if not actions:
            return Action.LISTEN
        if ood_score >= self.ood_threshold:
            return Action.LISTEN
        if risk >= self.high_risk:
            if Action.STOP in actions:
                return Action.STOP
            # Never execute an irreversible action solely because the learned
            # policy happened to be overconfident.
            return Action.CLARIFY
        if len(actions) == 1:
            return next(iter(actions))
        # Ambiguous interaction decisions resolve conservatively but without
        # treating benign acoustic uncertainty as an automatic refusal.
        for fallback in (Action.CLARIFY, Action.LISTEN, Action.YIELD, Action.STOP):
            if fallback in actions:
                return fallback
        if Action.EXECUTE in actions:
            return Action.CLARIFY
        return Action.LISTEN
