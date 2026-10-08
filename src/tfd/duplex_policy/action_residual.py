"""Zero-initialized correction of native listen/speak logits; frozen backbone."""
import torch
from torch import nn


class ActionResidual(nn.Module):
    def __init__(self, hidden_dim, rank=4):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, 2, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, native_logits):
        hidden = hidden.float()
        normalized = hidden / hidden.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return native_logits.float() + self.up(self.down(normalized))


def speaker_partition(speakers):
    names = sorted(set(speakers))
    if len(names) < 3:
        raise ValueError('at least three speakers required')
    held = names[::3]
    return [s for s in names if s not in held], held


def action_target(phase, policy):
    if policy not in {'ACK_IN_GAP', 'WAIT_RESUME'}:
        raise ValueError('unknown policy')
    if phase == 'boundary':
        return None
    if phase == 'speech':
        return 0
    if phase == 'post_final':
        return 1
    if phase == 'gap':
        return int(policy == 'ACK_IN_GAP')
    raise ValueError('unknown phase')
