from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from py_backend.tfd_runtime import (
    StreamingTrustGateController,
    StreamingTurnController,
    load_grpo_adapter,
)


class FakeLLM:
    def __init__(self):
        self.loaded = []
        self.active = []

    def load_adapter(self, path, adapter_name):
        self.loaded.append((path, adapter_name))

    def set_adapter(self, adapter_name):
        self.active = [adapter_name]

    def active_adapters(self):
        return self.active


class FakeModel:
    def __init__(self):
        self.llm = FakeLLM()


class FakeFeaturizer:
    def feed(self, frame):
        # Treat a positive test frame as voiced and zero as silence.
        energy_norm = 0.8 if float(np.max(np.abs(frame))) > 0.1 else 0.1
        return [energy_norm] + [0.0] * 9


class FakeStrategy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.current = "hold"

    def probs(self, _features):
        return {
            "hold": [0.9, 0.05, 0.05],
            "take": [0.05, 0.9, 0.05],
            "backchannel": [0.05, 0.05, 0.9],
        }[self.current]

    def feed(self, _features):
        self.current = next(self.actions)
        return self.current


class FakeGateDecision:
    action = "clarify"
    confidence = 0.4
    risk = 0.4
    score = 0.24
    rationale = "unit-test gate"
    confirmed = True


class FakeGate:
    def on_frame(self, **_kwargs):
        return FakeGateDecision()


def test_grpo_adapter_attaches_without_replacing_model(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    model = FakeModel()

    status = load_grpo_adapter(model, str(tmp_path), adapter_name="unit_test")

    assert status.loaded is True
    assert status.adapter_name == "unit_test"
    assert model.llm.loaded == [(str(tmp_path.resolve()), "unit_test")]
    assert model.llm.active == ["unit_test"]


def test_grpo_adapter_rejects_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        load_grpo_adapter(FakeModel(), str(tmp_path))


def test_turn_controller_waits_for_confirmed_take_and_real_speech():
    actions = ["hold", "hold", "take", "take"]
    controller = StreamingTurnController(
        featurizer_factory=FakeFeaturizer,
        strategy_factory=lambda: FakeStrategy(actions),
        frame_ms=100,
        min_voiced_frames=3,
        min_silence_frames=0,
    )
    frame = np.ones(1600, dtype=np.float32)

    assert controller.feed(frame).allow_speak is False
    assert controller.feed(frame).allow_speak is False
    assert controller.feed(frame).allow_speak is True


def test_turn_controller_keeps_partial_audio_until_full_frame():
    controller = StreamingTurnController(
        featurizer_factory=FakeFeaturizer,
        strategy_factory=lambda: FakeStrategy(["take"]),
        frame_ms=100,
        min_voiced_frames=1,
        min_silence_frames=0,
    )
    assert controller.feed(np.ones(800, dtype=np.float32)).processed_frames == 0
    decision = controller.feed(np.ones(800, dtype=np.float32))
    assert decision.processed_frames == 1
    assert decision.allow_speak is True


def test_turn_controller_accepts_read_only_audio_buffer():
    controller = StreamingTurnController(
        featurizer_factory=FakeFeaturizer,
        strategy_factory=lambda: FakeStrategy(["take"]),
        frame_ms=100,
        min_voiced_frames=1,
        min_silence_frames=0,
    )
    raw = np.ones(1600, dtype=np.float32).tobytes()
    read_only = np.frombuffer(raw, dtype=np.float32)

    decision = controller.feed(read_only)

    assert decision.processed_frames == 1
    assert decision.allow_speak is True


def test_turn_controller_reset_clears_previous_turn_state():
    controller = StreamingTurnController(
        featurizer_factory=FakeFeaturizer,
        strategy_factory=lambda: FakeStrategy(["take"]),
        frame_ms=100,
        min_voiced_frames=1,
        min_silence_frames=0,
    )
    assert controller.feed(np.ones(1600, dtype=np.float32)).allow_speak is True
    controller.reset()
    assert controller.processed_frames == 0
    assert controller.voiced_frames == 0
    assert controller.trailing_silence_frames == 0
    assert controller.last_action == "hold"


def test_trust_gate_uses_accumulated_real_audio_and_exposes_decision():
    observed = {}

    def transcribe(audio):
        observed["samples"] = audio.size
        return "帮我删除文件", -0.3, 0.7

    controller = StreamingTrustGateController(
        transcribe=transcribe,
        gate_factory=FakeGate,
    )
    controller.feed(np.ones(8000, dtype=np.float32))
    controller.feed(np.ones(8000, dtype=np.float32))

    decision = controller.evaluate()

    assert observed["samples"] == 16000
    assert decision.action == "clarify"
    assert decision.allow_execute is False
    assert decision.text == "帮我删除文件"
    assert decision.audio_seconds == 1.0


def test_trust_gate_discards_audio_older_than_configured_window():
    observed = {}

    def transcribe(audio):
        observed["samples"] = audio.size
        return "继续", -0.2, 0.8

    controller = StreamingTrustGateController(
        transcribe=transcribe,
        gate_factory=FakeGate,
        max_audio_seconds=1.0,
    )
    controller.feed(np.ones(8000, dtype=np.float32))
    controller.feed(np.ones(8000, dtype=np.float32))
    controller.feed(np.ones(8000, dtype=np.float32))

    controller.evaluate()

    # Whole chunks are retained, but the oldest chunk is dropped as soon as
    # the configured rolling window is exceeded.
    assert observed["samples"] == 16000


def test_empty_asr_result_keeps_listening_instead_of_canned_clarify():
    class LowConfidenceDecision:
        action = "clarify"
        confidence = 0.05
        risk = 0.0
        score = 0.0
        rationale = "confidence 0.05 < min 0.45"
        confirmed = True

    class LowConfidenceGate:
        def on_frame(self, **_kwargs):
            return LowConfidenceDecision()

    controller = StreamingTrustGateController(
        transcribe=lambda _audio: ("", -10.0, 0.0),
        gate_factory=LowConfidenceGate,
    )
    controller.feed(np.zeros(16000, dtype=np.float32))

    decision = controller.evaluate()

    assert decision.action == "listen"
    assert decision.listen_only is True
    assert decision.should_intercept is False


def test_low_risk_asr_uncertainty_defers_to_base_model():
    class LowConfidenceDecision:
        action = "clarify"
        confidence = 0.406
        risk = 0.0
        score = 0.0
        rationale = "confidence 0.41 < min 0.45"
        confirmed = True

    class LowConfidenceGate:
        def on_frame(self, **_kwargs):
            return LowConfidenceDecision()

    controller = StreamingTrustGateController(
        transcribe=lambda _audio: ("你是谁", -0.8, 0.4),
        gate_factory=LowConfidenceGate,
    )
    controller.feed(np.ones(16000, dtype=np.float32))

    decision = controller.evaluate()

    assert decision.action == "execute"
    assert decision.allow_execute is True
    assert decision.should_intercept is False
