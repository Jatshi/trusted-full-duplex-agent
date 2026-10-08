"""Non-invasive feature capture for MiniCPM-o's streaming decoder.

MiniCPM-o already computes both audio embeddings and causal LLM hidden states
inside ``StreamDecoder.feed``. The stock duplex wrapper discards the latter.
This context manager captures them without modifying model weights or caches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tfd.turntaking.features import FrameFeaturizer


@dataclass(frozen=True)
class StreamFeatureCapture:
    acoustic: torch.Tensor
    semantic: torch.Tensor


class StreamFeatureTap:
    def __init__(self, decoder: Any):
        if not hasattr(decoder, "feed"):
            raise TypeError("decoder must expose a feed method")
        self.decoder = decoder
        self._original = None
        self._armed = False
        self._captures: list[StreamFeatureCapture] = []

    def __enter__(self) -> "StreamFeatureTap":
        if self._original is not None:
            raise RuntimeError("feature tap is already installed")
        self._original = self.decoder.feed

        def wrapped(embeds: torch.Tensor, return_logits: bool = False):
            force_hidden = self._armed
            result = self._original(embeds, return_logits=(return_logits or force_hidden))
            if self._armed:
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError("decoder.feed did not return logits and hidden states")
                _, hidden = result
                self._captures.append(
                    StreamFeatureCapture(
                        acoustic=embeds.detach().float().cpu(),
                        semantic=hidden.detach().float().cpu().squeeze(0),
                    )
                )
            return result if return_logits else None

        self.decoder.feed = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._original is not None:
            self.decoder.feed = self._original
        self._original = None
        self._armed = False

    def arm(self) -> None:
        if self._original is None:
            raise RuntimeError("install the feature tap with a context manager before arming")
        self._captures.clear()
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def pop(self) -> StreamFeatureCapture:
        self._armed = False
        if not self._captures:
            raise RuntimeError("no feed call was captured")
        # In an audio+vision unit the final feed is the audio feed used to
        # produce pending logits, which is the state required by this policy.
        capture = self._captures[-1]
        self._captures.clear()
        return capture


def _pool_to_frames(features: torch.Tensor, frames: int) -> torch.Tensor:
    if features.ndim != 2 or not len(features) or frames <= 0:
        raise ValueError("features must be [tokens,dim] and frames must be positive")
    transposed = features.transpose(0, 1).unsqueeze(0)
    if features.shape[0] >= frames:
        pooled = F.adaptive_avg_pool1d(transposed, frames)
    else:
        pooled = F.interpolate(transposed, size=frames, mode="linear", align_corners=False)
    return pooled.squeeze(0).transpose(0, 1).contiguous()


def align_token_features(
    acoustic: torch.Tensor,
    semantic: torch.Tensor,
    *,
    duration_ms: int,
    frame_ms: int = 160,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map token clocks to frames for noncausal diagnostics only.

    Pooling an entire recording exposes future tokens to earlier frames. Do
    not use this helper for online policy training or timing evaluation.
    """
    if duration_ms <= 0 or frame_ms <= 0:
        raise ValueError("duration_ms and frame_ms must be positive")
    frames = max(1, (duration_ms + frame_ms - 1) // frame_ms)
    return _pool_to_frames(acoustic, frames), _pool_to_frames(semantic, frames)


def align_completed_chunks(
    captures: list[StreamFeatureCapture],
    *,
    chunk_ms: list[int],
    duration_ms: int,
    frame_ms: int = 160,
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """Expose each native chunk only after all its audio has arrived.

    MiniCPM-o duplex prefill consumes fixed 1035/1000 ms chunks. Its hidden
    state is unavailable during a chunk, even if a 160 ms training frame ends
    earlier. Before the first completed chunk the model gets zero features
    and an explicit availability mask. Partial final chunks are padded before
    prefill, so their completion time includes the padding duration.
    """
    if not captures or len(captures) != len(chunk_ms):
        raise ValueError("captures and chunk_ms must have the same nonzero length")
    if duration_ms <= 0 or frame_ms <= 0 or any(ms <= 0 for ms in chunk_ms):
        raise ValueError("duration_ms, frame_ms and all chunk lengths must be positive")

    acoustic_dim = captures[0].acoustic.shape[-1]
    semantic_dim = captures[0].semantic.shape[-1]
    for capture in captures:
        if capture.acoustic.ndim != 2 or capture.semantic.ndim != 2:
            raise ValueError("captured features must have shape [tokens, dim]")
        if not len(capture.acoustic) or not len(capture.semantic):
            raise ValueError("captured features cannot be empty")
        if capture.acoustic.shape[1] != acoustic_dim or capture.semantic.shape[1] != semantic_dim:
            raise ValueError("feature dimension drift between native chunks")

    acoustic_rows = []
    semantic_rows = []
    available = []
    completions = []
    elapsed = 0
    for ms in chunk_ms:
        elapsed += ms
        completions.append(elapsed)

    frames = (duration_ms + frame_ms - 1) // frame_ms
    visible_index = -1
    for frame_index in range(frames):
        frame_end_ms = min((frame_index + 1) * frame_ms, duration_ms)
        while visible_index + 1 < len(completions) and completions[visible_index + 1] <= frame_end_ms:
            visible_index += 1
        if visible_index < 0:
            acoustic_rows.append(torch.zeros(acoustic_dim, dtype=torch.float32))
            semantic_rows.append(torch.zeros(semantic_dim, dtype=torch.float32))
            available.append(False)
        else:
            acoustic_rows.append(captures[visible_index].acoustic.float().mean(0))
            semantic_rows.append(captures[visible_index].semantic.float()[-1])
            available.append(True)
    return torch.stack(acoustic_rows), torch.stack(semantic_rows), available


def causal_acoustic_frames(
    waveform: np.ndarray,
    *,
    frame_ms: int = 160,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Compute acoustic cues frame by frame from audio available so far."""
    if sample_rate != 16000:
        raise ValueError("FrameFeaturizer requires 16 kHz audio")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    samples = round(sample_rate * frame_ms / 1000)
    if samples <= 0:
        raise ValueError("frame_ms is too small for the sample rate")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("waveform must be a nonempty finite mono signal")
    featurizer = FrameFeaturizer(frame_ms=frame_ms)
    rows = []
    for start in range(0, len(audio), samples):
        frame = audio[start : start + samples]
        # Only the final incomplete frame is padded; no future audio exists.
        if len(frame) < samples:
            frame = np.pad(frame, (0, samples - len(frame)))
        rows.append(featurizer.feed(frame))
    return torch.tensor(rows, dtype=torch.float32)


def system_activity_frames(
    intervals: list[dict], *, duration_ms: int, frame_ms: int = 160
) -> torch.Tensor:
    """Playback state known by each frame end, never future playback state."""
    if duration_ms <= 0 or frame_ms <= 0 or not isinstance(intervals, list):
        raise ValueError("duration/frame must be positive and intervals must be a list")
    bounds = []
    for interval in intervals:
        start, end = float(interval["start_ms"]), float(interval["end_ms"])
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration_ms:
            raise ValueError("system active interval is outside recording duration")
        bounds.append((start, end))
    frames = (duration_ms + frame_ms - 1) // frame_ms
    return torch.tensor([
        [float(any(start < min((index + 1) * frame_ms, duration_ms)
                   and end > index * frame_ms for start, end in bounds))]
        for index in range(frames)
    ], dtype=torch.float32)
