"""Causal role-aware force-listen *rule baseline* for one-turn feasibility tests.

This is deliberately not a learned 3.0 method. It tests whether the existing
backend control channel can prevent a mid-story response without losing the
eventual answer. It sees only the currently arrived one-second PCM block and
has no access to manifest pause labels or future audio.
"""
from __future__ import annotations

import numpy as np


class CausalRoleFloorControl:
    """Choose the next input packet's force_listen bit from causal audio alone."""

    def __init__(self, policy_id: str, *, sample_rate: int = 16000,
                 frame_ms: int = 20, rms_threshold: float = 0.008,
                 gap_ms: int = 1500, resumption_ms: int = 200,
                 final_silence_ms: int = 1000, watchdog_ms: int = 5000) -> None:
        if policy_id not in {"ACK_IN_GAP", "WAIT_RESUME"}:
            raise ValueError(f"unknown policy: {policy_id}")
        if sample_rate != 16000 or frame_ms != 20 or not 0 < rms_threshold < 1:
            raise ValueError("unsupported input format or RMS threshold")
        if any(value <= 0 or value % frame_ms for value in
               (gap_ms, resumption_ms, final_silence_ms, watchdog_ms)):
            raise ValueError("durations must be positive multiples of frame_ms")
        self.policy_id = policy_id
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.packet_samples = sample_rate
        self.rms_threshold = rms_threshold
        self.gap_frames = gap_ms // frame_ms
        self.resumption_frames = resumption_ms // frame_ms
        self.final_silence_frames = final_silence_ms // frame_ms
        self.watchdog_frames = watchdog_ms // frame_ms
        self.speech_seen = False
        self.gap_seen = False
        self.resumption_seen = False
        self.release_reason: str | None = None
        self._silence_frames = 0
        self._resumption_voiced_frames = 0
        self._gap_elapsed_frames = 0
        self._final_silence_frames = 0
        self.processed_packets = 0

    def feed(self, audio_packet: np.ndarray) -> bool:
        """Return true to force listening for this just-arrived 1 s packet."""
        audio = np.asarray(audio_packet, dtype=np.float32)
        if (audio.ndim != 1 or audio.size != self.packet_samples
                or not np.isfinite(audio).all()):
            raise ValueError("expected finite mono Float32 one-second 16 kHz packet")
        self.processed_packets += 1
        if self.policy_id == "ACK_IN_GAP":
            return False
        if self.release_reason is not None:
            return False
        frames = audio.reshape(-1, self.frame_samples)
        voiced = np.sqrt(np.mean(np.square(frames), axis=1)) >= self.rms_threshold
        for active in voiced:
            if self.release_reason is not None:
                break
            if not self.speech_seen:
                self.speech_seen = bool(active)
                continue
            if not self.gap_seen:
                self._silence_frames = 0 if active else self._silence_frames + 1
                if self._silence_frames >= self.gap_frames:
                    self.gap_seen = True
                continue
            if not self.resumption_seen:
                self._gap_elapsed_frames += 1
                self._resumption_voiced_frames = (
                    self._resumption_voiced_frames + 1 if active else 0
                )
                if self._resumption_voiced_frames >= self.resumption_frames:
                    self.resumption_seen = True
                elif self._gap_elapsed_frames >= self.watchdog_frames:
                    self.release_reason = "no_resumption_watchdog"
                continue
            self._final_silence_frames = 0 if active else self._final_silence_frames + 1
            if self._final_silence_frames >= self.final_silence_frames:
                self.release_reason = "resumed_then_final_silence"
        return self.release_reason is None

    def snapshot(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "processed_packets": self.processed_packets,
            "speech_seen": self.speech_seen,
            "gap_seen": self.gap_seen,
            "resumption_seen": self.resumption_seen,
            "release_reason": self.release_reason,
        }
