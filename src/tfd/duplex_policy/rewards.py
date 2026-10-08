"""Factorized rewards and a minimal safety-constrained GRPO core.

The reward is deliberately auditable: every scalar corresponds to an outcome
that can be recomputed from a recorded duplex session. It is not an opaque
LLM-judge score and should not be confused with the language model's loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DuplexOutcome:
    expected_event: bool
    event_delay_ms: float | None
    task_success: bool
    unsafe_execute: bool
    false_takeover: bool
    inappropriate_backchannel: bool


@dataclass(frozen=True)
class RewardWeights:
    task: float = 1.0
    timing: float = 0.6
    false_takeover: float = 1.0
    safety: float = 2.0
    backchannel: float = 0.4
    latency_scale_ms: float = 500.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


def factorized_reward(
    outcome: DuplexOutcome,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    if weights.latency_scale_ms <= 0:
        raise ValueError("latency_scale_ms must be positive")
    task = weights.task if outcome.task_success else 0.0
    if outcome.expected_event:
        timing = (
            weights.timing * math.exp(-abs(float(outcome.event_delay_ms)) / weights.latency_scale_ms)
            if outcome.event_delay_ms is not None
            else -weights.timing
        )
    else:
        timing = 0.0
    false_takeover = -weights.false_takeover if outcome.false_takeover else 0.0
    safety = -weights.safety if outcome.unsafe_execute else 0.0
    backchannel = -weights.backchannel if outcome.inappropriate_backchannel else 0.0
    components = {
        "task": float(task),
        "timing": float(timing),
        "false_takeover": float(false_takeover),
        "safety": float(safety),
        "backchannel": float(backchannel),
    }
    return RewardBreakdown(total=float(sum(components.values())), components=components)


def group_relative_advantages(
    rewards: torch.Tensor,
    safety_costs: torch.Tensor,
    *,
    lagrange_multiplier: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute GRPO advantages after applying a Lagrangian safety cost.

    Inputs have shape ``[prompts, candidates]``. Normalisation occurs within
    each prompt group; no value network is used.
    """
    if rewards.ndim != 2 or safety_costs.shape != rewards.shape:
        raise ValueError("rewards and safety_costs must share [group,candidate] shape")
    if lagrange_multiplier < 0 or eps <= 0:
        raise ValueError("lagrange_multiplier must be non-negative and eps positive")
    adjusted = rewards - float(lagrange_multiplier) * safety_costs
    centered = adjusted - adjusted.mean(dim=1, keepdim=True)
    scale = adjusted.std(dim=1, keepdim=True, unbiased=False)
    return centered / scale.clamp_min(eps)


@dataclass
class ConstrainedGRPOState:
    """Dual variable for the constraint E[safety_cost] <= budget."""

    lagrange_multiplier: float = 0.0
    cost_budget: float = 0.05
    dual_lr: float = 0.01
    max_multiplier: float = 100.0

    def update(self, safety_costs: torch.Tensor) -> float:
        if safety_costs.numel() == 0 or not torch.isfinite(safety_costs).all():
            raise ValueError("safety_costs must be finite and non-empty")
        violation = float(safety_costs.float().mean().item()) - self.cost_budget
        updated = self.lagrange_multiplier + self.dual_lr * violation
        self.lagrange_multiplier = min(self.max_multiplier, max(0.0, updated))
        return self.lagrange_multiplier
