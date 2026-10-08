"""Experimental source-addressed recurrent evidence memory.

Not integrated with Demo; no novelty or performance claim. Revocation and source
addresses must be *observed causal protocol events*, never future labels or an
oracle near-end/echo identity. This module cannot certify those inputs.
"""
import torch
from torch import nn


class RevocableEvidenceMemory(nn.Module):
    """Learn writes/retention and WAIT/RESPOND/CANCEL logits; never force EOS.

    State is exclusively source-addressed, so revoking a source removes its
    recurrent contribution without a hidden global recurrent side channel.
    Immediate event features still reach the decision head: removing historical
    evidence is not a claim of removing all present input information.
    """

    def __init__(self, input_dim, hidden_dim, sources):
        super().__init__()
        if min(input_dim, hidden_dim, sources) <= 0:
            raise ValueError('dimensions must be positive')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sources = sources
        self.content = nn.Linear(input_dim, hidden_dim)
        self.write = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.retain = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.decision = nn.Linear(input_dim + sources * hidden_dim, 3)

    def initial_state(self, batch):
        # Begin a new response scope with a fresh state; no cross-response cache.
        return self.content.weight.new_zeros(batch, self.sources, self.hidden_dim)

    def step(self, event, source, state, revoke):
        batch = event.shape[0]
        if event.shape != (batch, self.input_dim):
            raise ValueError('event shape mismatch')
        if state.shape != (batch, self.sources, self.hidden_dim):
            raise ValueError('state shape mismatch')
        if source.shape != (batch,) or source.dtype != torch.long:
            raise ValueError('source must be a batch of integer addresses')
        if revoke.shape != (batch, self.sources) or revoke.dtype != torch.bool:
            raise ValueError('revoke must be a boolean source mask')
        if (source < 0).any() or (source >= self.sources).any():
            raise ValueError('source address out of range')
        if not torch.isfinite(event).all() or not torch.isfinite(state).all():
            raise ValueError('non-finite features/state')
        clean = state.masked_fill(revoke.unsqueeze(-1), 0)
        rows = torch.arange(batch, device=source.device)
        old = clean[rows, source]
        context = torch.cat((event, old), dim=-1)
        candidate = (torch.sigmoid(self.retain(context)) * old
                     + torch.sigmoid(self.write(context)) * torch.tanh(self.content(event)))
        # A revoked source cannot write in the same event. A later reauthorization
        # event can write anew; caller must retain revoke flags until that event.
        candidate = candidate.masked_fill(revoke[rows, source].unsqueeze(-1), 0)
        updated = clean.clone()
        updated[rows, source] = candidate
        logits = self.decision(torch.cat((event, updated.flatten(1)), dim=-1))
        return updated, logits
