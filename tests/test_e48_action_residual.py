import torch
from tfd.duplex_policy.action_residual import ActionResidual, speaker_partition, action_target


def test_zero_initialization_preserves_native_logits():
    head = ActionResidual(8)
    x, base = torch.randn(5, 8), torch.randn(5, 2)
    assert torch.equal(head(x, base), base)


def test_gradient_reaches_action_adapter_and_updates():
    head = ActionResidual(8)
    opt = torch.optim.AdamW(head.parameters(), lr=.01)
    x, base = torch.ones(4, 8), torch.zeros(4, 2)
    loss = torch.nn.functional.cross_entropy(head(x, base), torch.ones(4, dtype=torch.long))
    loss.backward()
    assert head.up.weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(head(x, base), base)


def test_speaker_partition_is_disjoint_and_order_invariant():
    speakers = ['a', 'b', 'c', 'd', 'e', 'f']
    train, test = speaker_partition(speakers)
    assert train and test and not set(train) & set(test)
    assert (train, test) == speaker_partition(list(reversed(speakers)))


def test_boundary_is_not_assigned_a_training_label():
    assert action_target('boundary', 'ACK_IN_GAP') is None
    assert action_target('gap', 'WAIT_RESUME') == 0
    assert action_target('gap', 'ACK_IN_GAP') == 1
    assert action_target('speech', 'ACK_IN_GAP') == 0
    assert action_target('post_final', 'WAIT_RESUME') == 1


def test_causal_vad_gate_keeps_partial_frames_and_requires_confirmation():
    import numpy as np
    from tfd.duplex_policy.causal_vad_gate import CausalVadGate
    class Mock:
        def reset_states(self):
            pass
        def __call__(self, audio, sr):
            assert sr == 16000 and len(audio) == 512
            return .9 if audio.mean() > .5 else .1
    gate = CausalVadGate(Mock())
    assert not gate.feed(np.ones(1024, dtype=np.float32))
    assert not gate.feed(np.ones(256, dtype=np.float32))
    assert gate.feed(np.ones(256, dtype=np.float32))
    gate.reset()
    assert not gate.feed(np.zeros(1536, dtype=np.float32))


def test_final_drain_requires_eof_and_new_response_then_natural_end():
    from tfd.duplex_policy.final_drain import FinalDrain
    state = FinalDrain()
    assert not state.observe(eof=False, is_listen=False, active_before=False, end_of_turn=True)
    assert not state.observe(eof=True, is_listen=False, active_before=True, end_of_turn=True)
    assert not state.observe(eof=True, is_listen=False, active_before=False, end_of_turn=False)
    assert state.observe(eof=True, is_listen=False, active_before=True, end_of_turn=True)
    assert state.started and state.complete
