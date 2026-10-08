"""Causal paired-channel auxiliary head for human-human turn-state learning."""
from __future__ import annotations

import torch
from torch import nn


class PairedTurnStateProbe(nn.Module):
    """Four-state prefix model with a same-capacity single-channel ablation.

    It does not emit the seven agent actions or bypass the TrustGate. The other
    stream is observed audio, never a fabricated agent playback log.
    """

    def __init__(self, hidden_dim: int = 64, num_states: int = 4):
        super().__init__()
        if hidden_dim <= 0 or not 2 <= num_states <= 4:
            raise ValueError("hidden_dim must be positive and num_states must be 2..4")
        self.target_encoder = nn.GRU(10, hidden_dim, batch_first=True)
        self.other_encoder = nn.GRU(11, hidden_dim, batch_first=True)
        self.other_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_states),
        )

    def forward(self, features: torch.Tensor, *, mode: str = "dual") -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 21:
            raise ValueError("features must have shape [batch,frames,21]")
        if mode not in ("single", "dual"):
            raise ValueError("mode must be 'single' or 'dual'")
        target, _ = self.target_encoder(features[:, :, :10])
        if mode == "single":
            fused = target
        else:
            other, _ = self.other_encoder(features[:, :, 10:])
            gate = torch.sigmoid(self.other_gate(torch.cat([target, other], dim=-1)))
            fused = target + gate * other
        return self.classifier(fused)
