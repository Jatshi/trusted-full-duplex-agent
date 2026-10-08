"""Causal four-state endpoint examples from public two-channel conversations.

These are human-human *turn-state* labels, not assistant action, safety, or
YIELD labels.  Keep this auxiliary task separate from SafeDuplex's seven-action
supervision and its formally calibrated split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .feature_tap import causal_acoustic_frames


TURN_STATES = ("complete", "incomplete", "backchannel", "wait")
TURN_STATE_TO_INDEX = {name: index for index, name in enumerate(TURN_STATES)}
SAMPLE_RATE = 16000
FRAME_MS = 160


def smoothconv_group_id(path: str | Path) -> str:
    """Keep all `*_seg...` cuts of one original conversation in one split."""
    normalized = str(path).replace("\\", "/")
    if not normalized.lower().endswith(".wav"):
        raise ValueError("SmoothConv audio path must end in .wav")
    stem = normalized[:-4]
    if "_seg" not in stem:
        raise ValueError("SmoothConv filename lacks the original conversation group")
    group_stem = stem.split("_seg", 1)[0]
    return "/".join(group_stem.split("/")[-2:])


def _load_two_channels(wav_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if audio.shape[1] != 2 or audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("public turn probe requires finite, two-channel audio")
    if sample_rate <= 0:
        raise ValueError("invalid sample rate")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor, axis=0
        ).astype(np.float32)
    return audio


def build_smoothconv_examples(
    wav_path: str | Path,
    annotation_path: str | Path,
    *,
    context_frames: int = 32,
    frame_ms: int = FRAME_MS,
    echo_gain: float = 0.0,
    echo_delay_ms: int = 80,
) -> dict:
    """One supervised window per annotated segment endpoint.

    The decision clock closes on the first frame ending at/after the annotated
    endpoint, so the allowed endpoint latency is < ``frame_ms``.  Each feature
    frame sees only audio up to its own end; future transcript/labels never
    enter the feature tensor.  The 21st feature is the *other human channel's*
    acoustic activity, not a real assistant playback log.
    """
    if (context_frames <= 0 or frame_ms <= 0 or not 0 <= echo_gain <= 1
            or echo_delay_ms < 0):
        raise ValueError("invalid context, frame clock, echo gain, or echo delay")
    wav_path = Path(wav_path)
    annotation_path = Path(annotation_path)
    audio = _load_two_channels(wav_path)
    duration = len(audio) / SAMPLE_RATE
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError("annotation must contain an instances list")

    channel_features = [
        causal_acoustic_frames(audio[:, channel], frame_ms=frame_ms).numpy()
        for channel in range(2)
    ]
    if channel_features[0].shape != channel_features[1].shape:
        raise RuntimeError("two-channel frame clocks differ")
    frame_count = len(channel_features[0])
    observed_features = channel_features
    if echo_gain > 0:
        delay_samples = round(echo_delay_ms * SAMPLE_RATE / 1000)
        observed_features = []
        for channel in range(2):
            echo = np.zeros(len(audio), dtype=np.float32)
            if delay_samples < len(audio):
                echo[delay_samples:] = audio[: len(audio) - delay_samples, 1 - channel]
            observed = np.clip(audio[:, channel] + echo_gain * echo, -1, 1)
            observed_features.append(
                causal_acoustic_frames(observed, frame_ms=frame_ms).numpy()
            )
    windows: list[np.ndarray] = []
    labels: list[int] = []
    channels: list[int] = []
    end_frames: list[int] = []
    for instance in instances:
        channel = instance.get("channelIndex")
        if isinstance(channel, bool) or channel not in (0, 1):
            raise ValueError("annotation channel must be 0 or 1")
        start, end = float(instance["start"]), float(instance["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.04):
            raise ValueError("annotation time is invalid or outside audio")
        state = str(instance.get("attributes", {}).get("turn", "")).lower()
        if state not in TURN_STATE_TO_INDEX:
            continue
        end_frame = min(frame_count - 1, max(0, math.ceil(end * 1000 / frame_ms) - 1))
        first = max(0, end_frame + 1 - context_frames)
        target = observed_features[channel][first : end_frame + 1]
        other = channel_features[1 - channel][first : end_frame + 1]
        # FrameFeaturizer's first component is normalized log-RMS; -38 dB is
        # its VAD threshold. This bit is based on observed other-channel audio.
        other_active = (other[:, :1] > (60 - 38) / 60).astype(np.float32)
        joined = np.concatenate([target, other, other_active], axis=1)
        window = np.zeros((context_frames, 21), dtype=np.float32)
        window[-len(joined) :] = joined
        windows.append(window)
        labels.append(TURN_STATE_TO_INDEX[state])
        channels.append(channel)
        end_frames.append(end_frame)
    if not windows:
        raise ValueError("annotation contains no supported four-state endpoint labels")
    return {
        "features": np.stack(windows),
        "labels": np.asarray(labels, dtype=np.int64),
        "channels": np.asarray(channels, dtype=np.int8),
        "end_frame": np.asarray(end_frames, dtype=np.int32),
        "group_id": smoothconv_group_id(wav_path),
        "duration_seconds": duration,
        "frame_ms": frame_ms,
        "label_scope": "human_human_endpoint_four_state",
        "echo_gain": echo_gain,
        "echo_delay_ms": echo_delay_ms,
    }
