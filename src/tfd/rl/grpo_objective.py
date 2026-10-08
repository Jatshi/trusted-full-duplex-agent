"""E46 candidate token-level objective; not wired into the legacy trainer.

Isolate numerical verification before authorizing any expensive GPU update.
Log-probabilities must be for the same sampled token at the same position.
"""
from __future__ import annotations

import torch


def completion_mask(ids: torch.Tensor, eos_id: int, lengths: torch.Tensor) -> torch.Tensor:
    """Keep valid generated tokens through first EOS (inclusive), never padding.

    Explicit lengths disambiguate a pad token equal to EOS. Zero-length rows
    are allowed here for diagnostics, but not by the optimization objective.
    """
    if ids.ndim != 2 or lengths.shape != (ids.shape[0],):
        raise ValueError('ids/lengths shape mismatch')
    if ids.dtype != torch.long or lengths.dtype != torch.long:
        raise ValueError('ids and lengths must be int64')
    if lengths.device != ids.device or bool(((lengths < 0) | (lengths > ids.shape[1])).any()):
        raise ValueError('invalid lengths')
    valid = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    eos = (ids == eos_id) & valid
    preceding_eos = eos.long().cumsum(dim=1) - eos.long()
    return valid & (preceding_eos == 0)


def token_grpo_loss(current: torch.Tensor, reference: torch.Tensor,
                    old: torch.Tensor, advantages: torch.Tensor,
                    mask: torch.Tensor, beta: float = 0.1,
                    clip_epsilon: float = 0.2) -> tuple[torch.Tensor, dict]:
    """Clipped per-token surrogate plus k3 reference penalty; sequence mean.

    Reference, old rollout logps and group advantages are fixed. This is a
    loss primitive, not a claim that the surrounding legacy trainer is fixed.
    """
    import math
    if (current.ndim != 2 or reference.shape != current.shape or old.shape != current.shape
            or mask.shape != current.shape or advantages.shape != (current.shape[0],)):
        raise ValueError('shape mismatch')
    if current.numel() == 0 or mask.dtype != torch.bool:
        raise ValueError('nonempty boolean mask required')
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError('invalid loss coefficients')
    if any(t.device != current.device for t in (reference, old, advantages, mask)):
        raise ValueError('device mismatch')
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('empty completion has no optimization target')
    if any(not bool(torch.isfinite(t).all()) for t in (current, reference, old, advantages)):
        raise ValueError('nonfinite inputs')
    # Erase padded positions before exponentiation, avoiding masked overflow.
    ratio = torch.exp(torch.where(mask, current - old.detach(), 0))
    delta = torch.where(mask, reference.detach() - current, 0)
    k3 = torch.exp(delta) - delta - 1
    adv = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * adv)
    token_loss = -surrogate + beta * k3
    if not bool(torch.isfinite(token_loss).all()):
        raise ValueError('nonfinite objective; do not silently clamp away instability')
    loss = ((token_loss * mask).sum(dim=1) / counts).mean()
    stats = {'mean_token_kl': float(((k3 * mask).sum(dim=1) / counts).mean().detach()),
             'valid_tokens': int(counts.sum()), 'sequences': current.shape[0],
             'objective': 'clipped_token_surrogate_k3_sequence_mean',
             'legacy_trainer_replaced': False}
    return loss, stats
