"""Optional Trusted Full-Duplex runtime components.

The upstream MiniCPM-o backend remains usable without the TFD repository.  The
learned turn-taking controller and GRPO adapter are enabled explicitly through
environment variables so a baseline and an enhanced runtime can be compared
with the same server code.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


logger = logging.getLogger("tfd.runtime")


_ASR_MODELS: dict[tuple[str, str, str], Any] = {}


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdapterStatus:
    enabled: bool
    loaded: bool
    path: str = ""
    adapter_name: str = ""
    error: str = ""

    def metrics(self) -> dict[str, Any]:
        return {f"tfd_grpo_{key}": value for key, value in asdict(self).items()}


def load_grpo_adapter(
    model: Any,
    adapter_path: Optional[str],
    *,
    adapter_name: str = "tfd_grpo",
) -> AdapterStatus:
    """Attach a PEFT adapter without wrapping/replacing the Qwen model class."""
    if not adapter_path:
        return AdapterStatus(enabled=False, loaded=False)

    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"TFD GRPO adapter directory not found: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"TFD GRPO adapter_config.json missing: {path}")

    llm = getattr(model, "llm", None)
    if llm is None or not hasattr(llm, "load_adapter"):
        raise TypeError("loaded MiniCPM-o model does not expose llm.load_adapter")

    llm.load_adapter(str(path), adapter_name=adapter_name)
    llm.set_adapter(adapter_name)
    active = llm.active_adapters() if callable(getattr(llm, "active_adapters", None)) else []
    if adapter_name not in active:
        raise RuntimeError(f"TFD GRPO adapter was loaded but is not active: {active}")

    logger.info("TFD GRPO adapter active: name=%s path=%s", adapter_name, path)
    return AdapterStatus(
        enabled=True,
        loaded=True,
        path=str(path),
        adapter_name=adapter_name,
    )


def load_grpo_adapter_from_env(model: Any) -> AdapterStatus:
    return load_grpo_adapter(
        model,
        os.environ.get("TFD_GRPO_ADAPTER", "").strip() or None,
        adapter_name=os.environ.get("TFD_GRPO_ADAPTER_NAME", "tfd_grpo").strip() or "tfd_grpo",
    )


@dataclass(frozen=True)
class TurnDecision:
    action: str
    allow_speak: bool
    probabilities: tuple[float, ...]
    processed_frames: int
    voiced_frames: int
    trailing_silence_frames: int

    def metrics(self) -> dict[str, Any]:
        return {
            "tfd_turn_action": self.action,
            "tfd_turn_allow_speak": self.allow_speak,
            "tfd_turn_prob_hold": round(self.probabilities[0], 4),
            "tfd_turn_prob_take": round(self.probabilities[1], 4),
            "tfd_turn_prob_backchannel": round(self.probabilities[2], 4),
            "tfd_turn_processed_frames": self.processed_frames,
            "tfd_turn_voiced_frames": self.voiced_frames,
            "tfd_turn_trailing_silence_frames": self.trailing_silence_frames,
        }


@dataclass(frozen=True)
class GuardrailDecision:
    action: str
    text: str
    confidence: float
    risk: float
    score: float
    rationale: str
    confirmed: bool
    asr_ms: float
    audio_seconds: float

    @property
    def allow_execute(self) -> bool:
        return self.action == "execute"

    @property
    def should_intercept(self) -> bool:
        return self.action in {"clarify", "stop"}

    @property
    def listen_only(self) -> bool:
        return self.action == "listen"

    def metrics(self) -> dict[str, Any]:
        return {
            "tfd_gate_enabled": True,
            "tfd_gate_action": self.action,
            "tfd_gate_text": self.text,
            "tfd_gate_confidence": round(self.confidence, 4),
            "tfd_gate_risk": round(self.risk, 4),
            "tfd_gate_score": round(self.score, 4),
            "tfd_gate_confirmed": self.confirmed,
            "tfd_gate_rationale": self.rationale,
            "tfd_gate_asr_ms": round(self.asr_ms, 1),
            "tfd_gate_audio_seconds": round(self.audio_seconds, 2),
        }


class StreamingTrustGateController:
    """Accumulate a real user turn, transcribe it, then apply TrustGate."""

    def __init__(
        self,
        *,
        transcribe: Callable[[np.ndarray], tuple[str, float, float]],
        gate_factory: Callable[[], Any],
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
    ) -> None:
        self._transcribe = transcribe
        self._gate_factory = gate_factory
        self.sample_rate = int(sample_rate)
        self.max_audio_samples = int(float(max_audio_seconds) * self.sample_rate)
        self.reset()

    def reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self.last_decision: Optional[GuardrailDecision] = None

    def feed(self, audio_waveform: np.ndarray) -> None:
        audio = np.array(audio_waveform, dtype=np.float32, copy=True).reshape(-1)
        if audio.size:
            self._chunks.append(np.nan_to_num(audio, copy=False))
            total = sum(chunk.size for chunk in self._chunks)
            while len(self._chunks) > 1 and total > self.max_audio_samples:
                total -= self._chunks.pop(0).size

    def evaluate(self, *, resume_context=None) -> GuardrailDecision:
        audio = (
            np.concatenate(self._chunks)
            if self._chunks
            else np.zeros(0, dtype=np.float32)
        )
        started = time.perf_counter()
        text, avg_logprob, asr_confidence = self._transcribe(audio)
        asr_ms = (time.perf_counter() - started) * 1000.0
        gate = self._gate_factory()
        decision = gate.on_frame(
            text=text,
            logprob=avg_logprob,
            asr_confidence=asr_confidence,
        )
        action = str(decision.action)
        rationale = str(decision.rationale)
        meaningful = "".join(char for char in text if char.isalnum())
        if not meaningful or (len(meaningful) <= 1 and float(decision.confidence) < 0.6):
            # Ambient noise and ASR non-speech must never trigger a canned
            # clarification.  The correct full-duplex behavior is to listen.
            action = "listen"
            rationale = "no reliable speech -> keep listening"
        elif (
            action == "clarify"
            and float(decision.risk) <= 0.0
            and rationale.startswith("confidence ")
        ):
            # This ASR is a safety monitor, not the conversational recognizer.
            # For low-risk speech, let the multimodal base model use the real
            # audio instead of replacing normal conversation with a fixed line.
            action = "execute"
            rationale = "low-risk ASR uncertainty -> defer to base model"
        if (resume_context is not None and rationale == "underspecified target -> clarify"
                and resume_context.resolve(text, action=action,
                                           risk=float(decision.risk),
                                           confidence=float(decision.confidence),
                                           min_confidence=max(.45, float(getattr(gate, "min_conf", .6))))):
            action = "execute"
            rationale = "exact conversational resume -> observed interrupted response"
        result = GuardrailDecision(
            action=action,
            text=text,
            confidence=float(decision.confidence),
            risk=float(decision.risk),
            score=float(decision.score),
            rationale=rationale,
            confirmed=bool(decision.confirmed),
            asr_ms=asr_ms,
            audio_seconds=float(audio.size) / self.sample_rate,
        )
        self.last_decision = result
        return result


class StreamingTurnController:
    """Feed arbitrary audio chunks into the trained 100 ms MLP controller."""

    HOLD = "hold"
    TAKE = "take"
    BACKCHANNEL = "backchannel"

    def __init__(
        self,
        *,
        featurizer_factory: Callable[[], Any],
        strategy_factory: Callable[[], Any],
        sample_rate: int = 16000,
        frame_ms: int = 100,
        min_voiced_frames: int = 3,
        min_silence_frames: int = 4,
        vad_db: float = -38.0,
    ) -> None:
        self._featurizer_factory = featurizer_factory
        self._strategy_factory = strategy_factory
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = int(round(self.sample_rate * self.frame_ms / 1000.0))
        self.min_voiced_frames = int(min_voiced_frames)
        self.min_silence_frames = int(min_silence_frames)
        self.vad_db = float(vad_db)
        self._buffer = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.featurizer = self._featurizer_factory()
        self.strategy = self._strategy_factory()
        self._buffer = np.zeros(0, dtype=np.float32)
        self.processed_frames = 0
        self.voiced_frames = 0
        self.trailing_silence_frames = 0
        self.last_action = self.HOLD
        self.last_probabilities = (1.0, 0.0, 0.0)

    def feed(self, audio_waveform: np.ndarray) -> TurnDecision:
        # Audio decoded from base64 commonly arrives through ``np.frombuffer``
        # and is therefore read-only.  ``nan_to_num(copy=False)`` may write in
        # place even when there are no visible NaNs, so normalize into an owned
        # writable array at the protocol boundary.
        audio = np.array(audio_waveform, dtype=np.float32, copy=True).reshape(-1)
        if audio.size:
            audio = np.nan_to_num(audio, copy=False)
            self._buffer = np.concatenate((self._buffer, audio))

        while self._buffer.size >= self.frame_samples:
            frame = self._buffer[: self.frame_samples]
            self._buffer = self._buffer[self.frame_samples :]
            features = self.featurizer.feed(frame)
            probabilities = tuple(float(x) for x in self.strategy.probs(features))
            if len(probabilities) != 3:
                raise RuntimeError(f"turn-taking strategy returned {len(probabilities)} probabilities")
            action = str(self.strategy.feed(features))
            if action not in {self.HOLD, self.TAKE, self.BACKCHANNEL}:
                raise RuntimeError(f"invalid turn-taking action: {action}")

            # Feature zero is energy dB normalized from [-60, 0] to [0, 1].
            energy_db = float(features[0]) * 60.0 - 60.0
            if energy_db > self.vad_db:
                self.voiced_frames += 1
                self.trailing_silence_frames = 0
            elif self.voiced_frames > 0:
                self.trailing_silence_frames += 1
            self.processed_frames += 1
            self.last_action = action
            self.last_probabilities = probabilities

        allow_speak = (
            self.last_action == self.TAKE
            and self.voiced_frames >= self.min_voiced_frames
            and self.trailing_silence_frames >= self.min_silence_frames
        )
        return TurnDecision(
            action=self.last_action,
            allow_speak=allow_speak,
            probabilities=self.last_probabilities,
            processed_frames=self.processed_frames,
            voiced_frames=self.voiced_frames,
            trailing_silence_frames=self.trailing_silence_frames,
        )


def build_turn_controller_from_env() -> Optional[StreamingTurnController]:
    if not env_enabled("TFD_TURN_MLP_ENABLED"):
        return None

    project_root = Path(
        os.environ.get("TFD_PROJECT_ROOT", "/root/autodl-tmp/trusted-full-duplex-agent")
    ).expanduser().resolve()
    source_root = project_root / "src"
    weights = Path(
        os.environ.get(
            "TFD_TURN_MLP_WEIGHTS",
            str(project_root / "outputs" / "turntaking" / "learned_mlp.pt"),
        )
    ).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"TFD source directory not found: {source_root}")
    if not weights.is_file():
        raise FileNotFoundError(f"TFD turn-taking weights not found: {weights}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from tfd.turntaking.features import FrameFeaturizer
    from tfd.turntaking.predictor import LearnedStrategy

    frame_ms = int(os.environ.get("TFD_TURN_FRAME_MS", "100"))
    confirm_frames = int(os.environ.get("TFD_TURN_CONFIRM_FRAMES", "4"))
    min_voiced_frames = int(os.environ.get("TFD_TURN_MIN_VOICED_FRAMES", "3"))
    min_silence_frames = int(os.environ.get("TFD_TURN_MIN_SILENCE_FRAMES", "4"))
    vad_db = float(os.environ.get("TFD_TURN_VAD_DB", "-38"))

    logger.info(
        "TFD turn MLP enabled: weights=%s frame_ms=%d confirm_frames=%d",
        weights,
        frame_ms,
        confirm_frames,
    )
    return StreamingTurnController(
        featurizer_factory=lambda: FrameFeaturizer(frame_ms=frame_ms, vad_db=vad_db),
        strategy_factory=lambda: LearnedStrategy(
            weights_path=str(weights), confirm_frames=confirm_frames
        ),
        frame_ms=frame_ms,
        min_voiced_frames=min_voiced_frames,
        min_silence_frames=min_silence_frames,
        vad_db=vad_db,
    )


def _shared_faster_whisper_model(model_path: Path, device: str, compute_type: str) -> Any:
    key = (str(model_path), device, compute_type)
    model = _ASR_MODELS.get(key)
    if model is not None:
        return model

    from faster_whisper import WhisperModel

    logger.info(
        "Loading TFD ASR: model=%s device=%s compute_type=%s",
        model_path,
        device,
        compute_type,
    )
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    _ASR_MODELS[key] = model
    return model


def build_trust_gate_from_env() -> Optional[StreamingTrustGateController]:
    """Build the real-waveform ASR -> TrustGate path when explicitly enabled."""

    if not env_enabled("TFD_TRUST_GATE_ENABLED"):
        return None

    project_root = Path(
        os.environ.get("TFD_PROJECT_ROOT", "/root/autodl-tmp/trusted-full-duplex-agent")
    ).expanduser().resolve()
    source_root = project_root / "src"
    config_path = Path(
        os.environ.get("TFD_GATE_CONFIG", str(project_root / "configs" / "gate.yaml"))
    ).expanduser().resolve()
    model_path = Path(
        os.environ.get(
            "TFD_ASR_MODEL",
            str(project_root / "weights" / "faster-whisper-small"),
        )
    ).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"TFD source directory not found: {source_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"TFD TrustGate config not found: {config_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"TFD ASR model not found: {model_path}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import yaml
    from tfd.gate.trust_gate import TrustGate

    with config_path.open("r", encoding="utf-8") as handle:
        gate_config = yaml.safe_load(handle)

    device = os.environ.get("TFD_ASR_DEVICE", "cuda").strip() or "cuda"
    default_compute = "float16" if device == "cuda" else "int8"
    compute_type = os.environ.get("TFD_ASR_COMPUTE_TYPE", default_compute).strip()
    model = _shared_faster_whisper_model(model_path, device, compute_type)
    try:
        from opencc import OpenCC

        zh_normalizer = OpenCC("t2s")
    except ImportError:
        zh_normalizer = None

    def transcribe(audio: np.ndarray) -> tuple[str, float, float]:
        if audio.size == 0:
            return "", -10.0, 0.0
        configured_language = os.environ.get("TFD_ASR_LANGUAGE", "zh")
        # faster-whisper detects language only when language is None.
        language = None if configured_language.strip().lower() == "auto" else configured_language
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=int(os.environ.get("TFD_ASR_BEAM_SIZE", "1")),
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        realized = list(segments)
        text = "".join(str(segment.text) for segment in realized).strip()
        if zh_normalizer is not None:
            text = zh_normalizer.convert(text)
        if not realized:
            return text, -10.0, 0.0
        durations = [max(float(segment.end) - float(segment.start), 1e-3) for segment in realized]
        total = sum(durations)
        avg_logprob = sum(
            float(segment.avg_logprob) * duration
            for segment, duration in zip(realized, durations)
        ) / total
        no_speech = sum(
            float(segment.no_speech_prob) * duration
            for segment, duration in zip(realized, durations)
        ) / total
        confidence = max(0.0, min(1.0, math.exp(avg_logprob) * (1.0 - no_speech)))
        return text, avg_logprob, confidence

    logger.info("TFD TrustGate enabled: config=%s asr=%s", config_path, model_path)
    return StreamingTrustGateController(
        transcribe=transcribe,
        gate_factory=lambda: TrustGate(gate_config),
        max_audio_seconds=float(os.environ.get("TFD_ASR_MAX_AUDIO_SECONDS", "12")),
    )
