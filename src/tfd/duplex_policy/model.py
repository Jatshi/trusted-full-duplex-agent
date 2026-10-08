"""Dual-timescale strictly causal policy for full-duplex speech decisions.

The policy is intentionally a small sidecar network.  It consumes acoustic
features at the real-time frame rate and semantic states produced by a frozen
speech-language backbone.  It does not claim to replace MiniCPM-o; it learns
the project's own timing, belief and selective-action policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DuplexPolicyConfig:
    acoustic_dim: int
    semantic_dim: int
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_multiplier: int = 4
    num_actions: int = 7
    num_risk_levels: int = 3
    dropout: float = 0.1
    max_frames: int = 2048

    def __post_init__(self) -> None:
        positive = {
            "acoustic_dim": self.acoustic_dim,
            "semantic_dim": self.semantic_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_actions": self.num_actions,
            "num_risk_levels": self.num_risk_levels,
            "max_frames": self.max_frames,
        }
        bad = {name: value for name, value in positive.items() if value <= 0}
        if bad:
            raise ValueError(f"policy dimensions must be positive: {bad}")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


class DuplexPolicyOutput(NamedTuple):
    action_logits: torch.Tensor
    take_hazard_logits: torch.Tensor
    yield_hazard_logits: torch.Tensor
    risk_alpha: torch.Tensor
    hidden: torch.Tensor


class CausalDuplexPolicy(nn.Module):
    """Fuse fast acoustics and slower semantics on one causal frame clock."""

    def __init__(self, config: DuplexPolicyConfig):
        super().__init__()
        self.config = config
        d = config.model_dim
        self.acoustic_projection = nn.Sequential(
            nn.LayerNorm(config.acoustic_dim),
            nn.Linear(config.acoustic_dim, d),
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, d),
        )
        self.fusion_gate = nn.Linear(2 * d, d)
        self.position = nn.Embedding(config.max_frames, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.ff_multiplier * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, config.num_actions)
        self.take_hazard_head = nn.Linear(d, 1)
        self.yield_hazard_head = nn.Linear(d, 1)
        self.risk_evidence_head = nn.Linear(d, config.num_risk_levels)

    def forward(
        self,
        acoustic: torch.Tensor,
        semantic: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> DuplexPolicyOutput:
        self._validate_inputs(acoustic, semantic, valid_mask)
        batch, frames, _ = acoustic.shape
        if valid_mask is None:
            valid_mask = torch.ones(batch, frames, dtype=torch.bool, device=acoustic.device)

        fast = self.acoustic_projection(acoustic)
        slow = self.semantic_projection(semantic)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([fast, slow], dim=-1)))
        fused = gate * fast + (1.0 - gate) * slow
        positions = torch.arange(frames, device=acoustic.device)
        fused = fused + self.position(positions).unsqueeze(0)

        # True entries above the diagonal are blocked.  This property is tested
        # explicitly because future leakage would invalidate every prefix claim.
        causal_mask = torch.triu(
            torch.ones(frames, frames, dtype=torch.bool, device=acoustic.device),
            diagonal=1,
        )
        hidden = self.temporal(
            fused,
            mask=causal_mask,
            src_key_padding_mask=~valid_mask,
        )
        hidden = self.final_norm(hidden)
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        action_logits = self.action_head(hidden)
        take_logits = self.take_hazard_head(hidden).squeeze(-1)
        yield_logits = self.yield_hazard_head(hidden).squeeze(-1)
        # Dirichlet concentration alpha=evidence+1 is strictly >1 and gives an
        # explicit epistemic-strength signal sum(alpha).
        risk_alpha = F.softplus(self.risk_evidence_head(hidden)) + 1.0
        return DuplexPolicyOutput(
            action_logits=action_logits,
            take_hazard_logits=take_logits,
            yield_hazard_logits=yield_logits,
            risk_alpha=risk_alpha,
            hidden=hidden,
        )

    def _validate_inputs(
        self,
        acoustic: torch.Tensor,
        semantic: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if acoustic.ndim != 3 or semantic.ndim != 3:
            raise ValueError("acoustic and semantic inputs must have shape [batch, time, dim]")
        if acoustic.shape[:2] != semantic.shape[:2]:
            raise ValueError("acoustic and semantic streams must share batch/time dimensions")
        if acoustic.shape[-1] != self.config.acoustic_dim:
            raise ValueError("unexpected acoustic feature dimension")
        if semantic.shape[-1] != self.config.semantic_dim:
            raise ValueError("unexpected semantic feature dimension")
        if acoustic.shape[1] > self.config.max_frames:
            raise ValueError("sequence exceeds configured max_frames")
        if valid_mask is not None:
            if valid_mask.shape != acoustic.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool with shape [batch, time]")
            if not valid_mask[:, 0].all():
                raise ValueError("every sequence must contain at least one valid frame")
