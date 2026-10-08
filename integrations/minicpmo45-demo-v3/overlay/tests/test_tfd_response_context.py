"""Synthetic safety boundaries; not speech recognition or model quality evidence."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from py_backend.tfd_response_context import InterruptedResponseContext
from py_backend.tfd_runtime import StreamingTrustGateController


def evaluate(text, context, *, risk=0., confidence=.8, action='clarify'):
    decision = SimpleNamespace(action=action, confidence=confidence, risk=risk,
                               score=0., rationale='underspecified target -> clarify', confirmed=True)
    controller = StreamingTrustGateController(
        transcribe=lambda _: (text, -.1, confidence),
        gate_factory=lambda: SimpleNamespace(on_frame=lambda **_: decision))
    controller.feed(np.zeros(1600, dtype=np.float32))
    return controller.evaluate(resume_context=context)


def interrupted(clock=None, safe=True):
    context = InterruptedResponseContext(clock=clock or (lambda: 10.))
    context.begin('response_a', safe=safe)
    context.output_text('故事开始了')
    context.cancel('response_a')
    return context


@pytest.mark.parametrize('text', ['继续刚才那个故事', '请接着讲你刚才说的那个故事。', '继续讲刚才的内容'])
def test_exact_resume_requires_observed_interrupted_safe_response(text):
    context = interrupted()
    assert evaluate(text, context).action == 'execute'
    assert context.metrics()['tfd_resume_resolved_count'] == 1
    assert evaluate(text, context).action == 'clarify'  # single-use capability


@pytest.mark.parametrize('text', ['把那个删掉', '继续刚才那个故事，然后转账', '继续执行那个操作',
                                '那个你看着办', '继续刚才那个故事，开门', '继续刚才那个预约'])
def test_other_ambiguous_or_operational_requests_never_bypass(text):
    assert evaluate(text, interrupted()).action == 'clarify'


@pytest.mark.parametrize('risk,confidence,action', [(0.4,.9,'clarify'), (0.,.5,'clarify'),
                                                  (0.,math.nan,'clarify'), (math.nan,.9,'clarify'),
                                                  (0.,.9,'stop')])
def test_risk_confidence_and_stop_remain_authoritative(risk, confidence, action):
    assert evaluate('继续刚才那个故事', interrupted(), risk=risk,
                    confidence=confidence, action=action).action == action


def test_no_context_unsafe_or_expired_context_is_not_authorization():
    now = [10.]
    expired = interrupted(lambda: now[0])
    now[0] = 131.
    for context in [None, InterruptedResponseContext(), interrupted(safe=False), expired]:
        assert evaluate('继续刚才那个故事', context).action == 'clarify'


def test_response_ownership_output_and_new_turn_clear_context():
    context = InterruptedResponseContext(clock=lambda: 10.)
    context.begin('a', safe=True)
    context.cancel('a')  # no output, nothing to resume
    assert evaluate('继续刚才那个故事', context).action == 'clarify'
    context.begin('a', safe=True)
    context.output_text('故事')
    context.cancel('wrong')
    assert evaluate('继续刚才那个故事', context).action == 'clarify'
    context.cancel('a')
    context.begin('b', safe=True)  # intervening response invalidates old context
    assert evaluate('继续刚才那个故事', context).action == 'clarify'


def test_actual_trust_gate_ambiguity_rationale_and_baseline_unchanged():
    import sys
    from pathlib import Path
    import yaml
    root = Path(os.environ.get('TFD_PROJECT_ROOT', str(Path(__file__).parents[2] / 'trusted-full-duplex-agent')))
    sys.path.insert(0, str(root / 'src'))
    from tfd.gate.trust_gate import TrustGate
    config = yaml.safe_load((root / 'configs/gate.yaml').read_text(encoding='utf-8'))
    text = '请接着讲你刚才说的那个故事。'
    # Existing calibrated gate accepts ~.478 confidence; an arbitrary .6
    # resume threshold would leave the actual Whisper path overblocked.
    controller = StreamingTrustGateController(transcribe=lambda _: (text, -.45, .95),
                                             gate_factory=lambda: TrustGate(config))
    assert controller.evaluate().action == 'clarify'
    assert controller.evaluate(resume_context=interrupted()).action == 'execute'
