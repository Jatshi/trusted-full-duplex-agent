"""Small, auditable trainer for the causal policy sidecar."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from .data import DuplexBatch
from .losses import DuplexLossWeights, duplex_policy_loss
from .model import CausalDuplexPolicy, DuplexPolicyConfig


class DuplexPolicyTrainer:
    def __init__(
        self,
        model: CausalDuplexPolicy,
        *,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
        device: str | torch.device | None = None,
        loss_weights: DuplexLossWeights = DuplexLossWeights(),
    ) -> None:
        if lr <= 0 or weight_decay < 0 or grad_clip <= 0:
            raise ValueError("invalid optimizer hyperparameters")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.loss_weights = loss_weights

    def train_epoch(self, batches: Iterable[DuplexBatch]) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = defaultdict(float)
        steps = 0
        for raw_batch in batches:
            batch = raw_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch.acoustic, batch.semantic, batch.valid_mask)
            loss, parts = duplex_policy_loss(
                output,
                actions=batch.actions,
                risk_labels=batch.risk_labels,
                take_event_index=batch.take_event_index,
                yield_event_index=batch.yield_event_index,
                valid_mask=batch.valid_mask,
                weights=self.loss_weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite duplex policy loss: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["grad_norm"] += float(grad_norm.detach().cpu())
            for name, value in parts.items():
                totals[name] += float(value.detach().cpu())
            steps += 1
        if steps == 0:
            raise ValueError("train_epoch received no batches")
        return {name: value / steps for name, value in sorted(totals.items())}

    @torch.no_grad()
    def evaluate(self, batches: Iterable[DuplexBatch]) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = defaultdict(float)
        steps = 0
        correct = 0
        frames = 0
        for raw_batch in batches:
            batch = raw_batch.to(self.device)
            output = self.model(batch.acoustic, batch.semantic, batch.valid_mask)
            loss, parts = duplex_policy_loss(
                output,
                actions=batch.actions,
                risk_labels=batch.risk_labels,
                take_event_index=batch.take_event_index,
                yield_event_index=batch.yield_event_index,
                valid_mask=batch.valid_mask,
                weights=self.loss_weights,
            )
            totals["loss"] += float(loss.cpu())
            for name, value in parts.items():
                totals[name] += float(value.cpu())
            predictions = output.action_logits.argmax(-1)
            correct += int(((predictions == batch.actions) & batch.valid_mask).sum().item())
            frames += int(batch.valid_mask.sum().item())
            steps += 1
        if steps == 0 or frames == 0:
            raise ValueError("evaluate received no valid frames")
        result = {name: value / steps for name, value in sorted(totals.items())}
        result["action_accuracy"] = correct / frames
        return result


def save_checkpoint(
    path: str | Path,
    *,
    model: CausalDuplexPolicy,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "tfd-star-duplex-policy-v3",
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CausalDuplexPolicy, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != "tfd-star-duplex-policy-v3":
        raise ValueError("unsupported duplex-policy checkpoint format")
    model = CausalDuplexPolicy(DuplexPolicyConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model, payload
