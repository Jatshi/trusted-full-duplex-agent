"""Losses for event timing, prefix stability, actions and risk calibration."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import DuplexPolicyOutput


def discrete_time_hazard_nll(
    hazard_logits: torch.Tensor,
    event_index: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for one event time or right censoring.

    ``event_index=-1`` means the event was not observed before the sequence
    ended.  Frames after an observed event are ignored rather than treated as
    negative samples.
    """
    if hazard_logits.shape != valid_mask.shape:
        raise ValueError("hazard_logits and valid_mask must have identical shapes")
    if event_index.shape != (hazard_logits.shape[0],):
        raise ValueError("event_index must have shape [batch]")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")

    losses = []
    for row, event, mask in zip(hazard_logits, event_index, valid_mask):
        length = int(mask.sum().item())
        if length <= 0:
            raise ValueError("empty sequence is not valid for survival loss")
        idx = int(event.item())
        if idx >= length:
            raise ValueError(f"event index {idx} is outside valid sequence length {length}")
        if idx < -1:
            raise ValueError("event index must be -1 or non-negative")
        if idx == -1:
            # -log(1-sigmoid(x)) = softplus(x)
            losses.append(F.softplus(row[:length]).sum())
        else:
            survival = F.softplus(row[:idx]).sum()
            event_loss = F.softplus(-row[idx])
            losses.append(survival + event_loss)
    return torch.stack(losses).mean()


def prefix_consistency_kl(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between consecutive causal belief/action distributions."""
    if logits.ndim != 3 or valid_mask.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,C] and mask [B,T]")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    log_p = F.log_softmax(logits[:, 1:], dim=-1)
    log_q = F.log_softmax(logits[:, :-1], dim=-1)
    p, q = log_p.exp(), log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(-1)
    kl_qp = (q * (log_q - log_p)).sum(-1)
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    if not pair_mask.any():
        return logits.sum() * 0.0
    return (0.5 * (kl_pq + kl_qp))[pair_mask].mean()


def masked_action_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels.shape != logits.shape[:2] or valid_mask.shape != labels.shape:
        raise ValueError("action labels/mask must match [batch, time]")
    return F.cross_entropy(logits[valid_mask], labels[valid_mask], weight=class_weights)


def evidential_risk_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    evidence_penalty: float = 1e-3,
) -> torch.Tensor:
    """Dirichlet expected NLL plus a small wrong-evidence regularizer."""
    if labels.shape != alpha.shape[:2] or valid_mask.shape != labels.shape:
        raise ValueError("risk labels/mask must match [batch, time]")
    a = alpha[valid_mask]
    y = labels[valid_mask]
    strength = a.sum(-1)
    expected_nll = torch.digamma(strength) - torch.digamma(a.gather(1, y[:, None]).squeeze(1))
    one_hot = F.one_hot(y, num_classes=a.shape[-1]).to(a.dtype)
    wrong_evidence = ((a - 1.0) * (1.0 - one_hot)).sum(-1)
    return (expected_nll + evidence_penalty * wrong_evidence).mean()


@dataclass(frozen=True)
class DuplexLossWeights:
    action: float = 1.0
    take_hazard: float = 0.7
    yield_hazard: float = 0.5
    risk: float = 0.8
    prefix_consistency: float = 0.05


def duplex_policy_loss(
    output: DuplexPolicyOutput,
    *,
    actions: torch.Tensor,
    risk_labels: torch.Tensor,
    take_event_index: torch.Tensor,
    yield_event_index: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: DuplexLossWeights = DuplexLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = {
        "action": masked_action_cross_entropy(output.action_logits, actions, valid_mask),
        "take_hazard": discrete_time_hazard_nll(
            output.take_hazard_logits, take_event_index, valid_mask
        ),
        "yield_hazard": discrete_time_hazard_nll(
            output.yield_hazard_logits, yield_event_index, valid_mask
        ),
        "risk": evidential_risk_loss(output.risk_alpha, risk_labels, valid_mask),
        "prefix_consistency": prefix_consistency_kl(output.action_logits, valid_mask),
    }
    total = (
        weights.action * parts["action"]
        + weights.take_hazard * parts["take_hazard"]
        + weights.yield_hazard * parts["yield_hazard"]
        + weights.risk * parts["risk"]
        + weights.prefix_consistency * parts["prefix_consistency"]
    )
    return total, parts
