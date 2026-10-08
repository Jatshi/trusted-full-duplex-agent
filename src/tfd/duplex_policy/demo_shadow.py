"""Real trained action-head observation, never mutating logits or cancelling audio."""
from pathlib import Path
import time
import torch

from tfd.duplex_policy.demo_lora import BASE_PROFILES, LORA_PROFILES
PROFILES = BASE_PROFILES + LORA_PROFILES


def validate_profile(value):
    if value not in PROFILES:
        raise ValueError('Unknown/unsafe TFD Demo profile')
    return value


class ActionShadow:
    def __init__(self, duplex, heads, *, hidden_dim):
        self.duplex = duplex
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.applied_chunks = 0
        self.predictions = {}
        self.native_prediction = None
        self.head_wall_ms = {}

    def prefill(self, operation):
        self.predictions = {}
        self.native_prediction = None
        self.head_wall_ms = {}
        decoder = self.duplex.decoder
        original = decoder.feed
        hidden = None

        def capture(embeds, return_logits=False):
            nonlocal hidden
            result = original(embeds, return_logits=True)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError('No causal decoder hidden available')
            hidden = result[1].reshape(-1, result[1].shape[-1])[-1].detach()
            return result if return_logits else None

        decoder.feed = capture
        try:
            result = operation()
        finally:
            decoder.feed = original
        if hidden is None or hidden.shape != (self.hidden_dim,) or not torch.isfinite(hidden).all():
            raise ValueError('Invalid fresh causal hidden')
        ids = [self.duplex.listen_token_id, self.duplex.speak_token_id]
        pending = self.duplex.pending_logits
        if pending is None:
            raise ValueError('No native action logits')
        pair = pending.reshape(-1, pending.shape[-1])[-1, ids].float().clone()
        if not torch.isfinite(pair).all():
            raise ValueError('Nonfinite native logits')
        labels = ('LISTEN', 'SPEAK')
        with torch.no_grad():
            self.native_prediction = labels[int(pair.argmax())]
            for name, head, mode, _sha in self.heads:
                started = time.perf_counter()
                predicted = head(hidden, torch.zeros_like(pair) if mode == 'direct' else pair)
                if predicted.shape != (2,) or not torch.isfinite(predicted).all():
                    raise ValueError('Invalid shadow prediction')
                self.predictions[name] = labels[int(predicted.argmax())]
                # finite/argmax CPU reads synchronize CUDA. This is observed
                # wall time including synchronization, not isolated kernel time.
                self.head_wall_ms[name] = round((time.perf_counter() - started) * 1000, 3)
        self.applied_chunks += 1
        return result

    def metrics(self):
        return {'tfd_shadow_enabled': True, 'tfd_shadow_controls_output': False,
                'tfd_shadow_applied_chunks': self.applied_chunks,
                'tfd_shadow_native_prediction': self.native_prediction,
                'tfd_shadow_predictions': dict(self.predictions),
                'tfd_shadow_head_wall_ms': dict(self.head_wall_ms),
                'tfd_shadow_timing_scope': 'head_forward_finite_argmax_sync_wall_not_kernel_or_e2e',
                'tfd_shadow_checkpoint_sha256': {name: sha for name, _, _, sha in self.heads}}


def load_action_shadow(duplex, directory):
    from tfd.duplex_policy.native_action_bridge import load_native_action_bridge
    root = Path(directory).resolve(strict=True)
    heads, dimensions = [], set()
    # All residual seeds, and the existing equal-budget direct control. No best-seed selection.
    for name, mode in [('action_residual_seed42', 'residual'), ('action_residual_seed43', 'residual'),
                       ('action_residual_seed44', 'residual'), ('direct_action_head_seed42', 'direct')]:
        bridge = load_native_action_bridge(duplex, root / (name + '.pt'), mode=mode)
        dimensions.add(bridge.hidden_dim)
        heads.append((name, bridge.head, mode, bridge.checkpoint_sha256))
    if len(dimensions) != 1:
        raise ValueError('Shadow head dimensions disagree')
    return ActionShadow(duplex, heads, hidden_dim=dimensions.pop())
