"""Clipped group-relative policy objective for duplex action trajectories."""
from __future__ import annotations

import torch


def clipped_grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    reference_log_probs: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO-style clipped loss using group-relative, critic-free advantages."""
    if not (
        new_log_probs.shape == old_log_probs.shape == advantages.shape == valid_mask.shape
    ):
        raise ValueError("all GRPO tensors must have the same shape")
    if valid_mask.dtype != torch.bool or not valid_mask.any():
        raise ValueError("valid_mask must be boolean and contain a valid item")
    if not 0 < clip_epsilon < 1 or kl_coefficient < 0:
        raise ValueError("invalid GRPO clipping or KL coefficient")
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped)[valid_mask].mean()
    approx_kl = ((ratio - 1.0) - torch.log(ratio))[valid_mask].mean()
    reference_kl = new_log_probs.new_zeros(())
    if reference_log_probs is not None:
        if reference_log_probs.shape != new_log_probs.shape:
            raise ValueError("reference_log_probs must match candidate log probabilities")
        # Sampled reverse-KL proxy, non-negative near the reference policy.
        delta = reference_log_probs - new_log_probs
        reference_kl = (torch.exp(delta) - delta - 1.0)[valid_mask].mean()
    total = policy_loss + kl_coefficient * reference_kl
    return total, {
        "policy_loss": policy_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "reference_kl": reference_kl.detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_epsilon)[valid_mask].float().mean().detach(),
    }
