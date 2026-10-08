"""Opt-in native action correction at the prefill/generate boundary.

Only two pending action logits are modified. No cancellation, VAD, EOF,
response-completion or text-generation policy is inferred here. The caller
must serialize prefill/generate/finalize on the shared model, as in 2.0.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tfd.duplex_policy.action_residual import ActionResidual


class NativeActionBridge:
    def __init__(self, duplex, head, *, hidden_dim, mode="residual"):
        if mode not in {"residual", "direct"}:
            raise ValueError("unknown action head mode")
        self.duplex = duplex
        self.head = head
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.applied_chunks = 0
        self.checkpoint_sha256 = ""
        self.ids = [int(duplex.listen_token_id), int(duplex.speak_token_id)]
        if self.hidden_dim <= 0 or min(self.ids) < 0 or len(set(self.ids)) != 2:
            raise ValueError("invalid dimension or action token IDs")

    def prefill(self, operation):
        """Capture the last causal hidden on-device; always restore decoder.feed."""
        decoder = self.duplex.decoder
        original = decoder.feed
        hidden = None

        def capture(embeds, return_logits=False):
            nonlocal hidden
            result = original(embeds, return_logits=True)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("decoder did not return hidden capture")
            # Keep one detached vector, not every token or a CPU copy. This
            # avoids feature-dump D2H traffic in the real-time runtime.
            states = result[1]
            hidden = states.reshape(-1, states.shape[-1])[-1].detach()
            return result if return_logits else None

        decoder.feed = capture
        try:
            result = operation()
        finally:
            decoder.feed = original
        if hidden is None:
            raise RuntimeError("no fresh hidden capture in prefill")
        if hidden.shape != (self.hidden_dim,):
            raise ValueError("hidden dimension mismatch")
        pending = self.duplex.pending_logits
        if pending is None or max(self.ids) >= pending.shape[-1]:
            raise ValueError("pending action logits unavailable")
        if not pending.is_contiguous():
            raise ValueError("pending logits must be contiguous for in-place correction")
        vector = pending.view(-1, pending.shape[-1])[-1]
        pair = vector[self.ids].float()
        if not torch.isfinite(hidden).all() or not torch.isfinite(pair).all():
            raise ValueError("features and native logits must be finite")
        with torch.no_grad():
            if self.mode == "direct":
                learned = self.head(hidden, torch.zeros_like(pair))
                corrected = pair.mean() + learned - learned.mean()
            else:
                corrected = self.head(hidden, pair)
            cast = corrected.to(device=vector.device, dtype=vector.dtype)
            if cast.shape != (2,) or not torch.isfinite(cast).all():
                raise ValueError("head output must be two finite logits")
            vector[self.ids] = cast
        self.applied_chunks += 1
        return result

    def metrics(self):
        return {"tfd_native_head_enabled": True, "tfd_native_head_mode": self.mode,
                "tfd_native_head_applied_chunks": self.applied_chunks,
                "tfd_native_head_sha256": self.checkpoint_sha256}


def load_native_action_bridge(duplex, checkpoint, *, mode="residual"):
    """Load local project weights only; never silently fall back when enabled."""
    path = Path(checkpoint).expanduser().resolve(strict=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["action_token_ids"] != [duplex.listen_token_id, duplex.speak_token_id]:
        raise ValueError("checkpoint action token IDs do not match runtime")
    # StreamDecoder is a wrapper, not an nn.Module. Load before first prefill:
    # pending_logits is normally None after prepare; use the capability device.
    device = getattr(duplex, "device", None)
    if device is None:
        if duplex.pending_logits is None:
            raise ValueError("action head requires a runtime device")
        device = duplex.pending_logits.device
    head = ActionResidual(state["hidden_dim"], state["rank"]).to(device).eval()
    head.load_state_dict(state["state_dict"], strict=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    bridge = NativeActionBridge(duplex, head, hidden_dim=state["hidden_dim"], mode=mode)
    bridge.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return bridge
