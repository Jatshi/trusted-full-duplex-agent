"""Playback-reference near-end activity probe on the AEC Challenge corpus.

The clean near-end and isolated echo tracks are used *only* for offline labels
and integrity checks.  Inference inputs are microphone mixture + known
far-end playback reference.  This solves source attribution, not floor intent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from .feature_tap import causal_acoustic_frames


FRAME_MS = 40
SAMPLE_RATE = 16000
TRACK_FILES = {
    "nearend_speech": "nearend_speech_fileid_{id}.wav",
    "farend_speech": "farend_speech_fileid_{id}.wav",
    "echo": "echo_fileid_{id}.wav",
    "nearend_mic": "nearend_mic_fileid_{id}.wav",
}


def _bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 4


def select_speaker_disjoint_pilot(
    rows: list[dict], *, train_count: int, val_count: int, seed: int = 42,
) -> dict[str, list[dict]]:
    """Reserve official test untouched; isolate both near/far speaker identities."""
    if train_count < 1 or val_count < 1:
        raise ValueError("pilot train and validation sizes must be positive")
    pools: dict[str, list[dict]] = {"train": [], "val": []}
    for row in rows:
        if row["split"] != "train":
            continue
        near = _bucket(str(row["nearend_speaker"]), seed)
        far = _bucket(str(row["farend_speaker"]), seed)
        if near == 0 and far == 0:
            pools["val"].append(row)
        elif near != 0 and far != 0:
            pools["train"].append(row)
    result = {}
    for part, count in (("train", train_count), ("val", val_count)):
        ranked = sorted(pools[part], key=lambda row: hashlib.sha256(
            f"{seed}:{row['fileid']}".encode()).digest())
        if len(ranked) < count:
            raise ValueError(f"not enough speaker-disjoint {part} examples")
        result[part] = ranked[:count]
    train_speakers = {str(row[key]) for row in result["train"]
                      for key in ("nearend_speaker", "farend_speaker")}
    val_speakers = {str(row[key]) for row in result["val"]
                    for key in ("nearend_speaker", "farend_speaker")}
    if train_speakers & val_speakers:
        raise AssertionError("speaker-disjoint partition failed")
    return result


def aec_frame_record(
    directory: str | Path,
    fileid: str | int,
    *,
    nearend_scale: float,
    is_nearend_noisy: bool = False,
) -> dict[str, np.ndarray | float]:
    """Create 40 ms causal inputs and offline clean-near-end activity labels."""
    if not np.isfinite(nearend_scale) or nearend_scale <= 0:
        raise ValueError("nearend_scale must be finite and positive")
    directory = Path(directory)
    tracks = {}
    for name, pattern in TRACK_FILES.items():
        audio, sample_rate = sf.read(directory / pattern.format(id=fileid),
                                     dtype="float32", always_2d=True)
        if (sample_rate != SAMPLE_RATE or audio.shape[1] != 1 or audio.shape[0] == 0
                or not np.isfinite(audio).all()):
            raise ValueError(f"invalid mono 16 kHz AEC track: {name}")
        tracks[name] = audio[:, 0]
    lengths = {len(audio) for audio in tracks.values()}
    if len(lengths) != 1:
        raise ValueError("AEC tracks have mismatched sample clocks")
    near = tracks["nearend_speech"] * nearend_scale
    mic = tracks["nearend_mic"]
    echo = tracks["echo"]
    residual_rms = float(np.sqrt(np.mean((mic - near - echo) ** 2)))
    if not is_nearend_noisy:
        # The official PCM tracks include a few locally saturated mixtures;
        # the largest clean-row residual in this pinned pilot is 0.01056.
        # Reject grossly inconsistent tracks, but preserve/report that row.
        if residual_rms > 0.02:
            raise ValueError("AEC microphone mixture disagrees with clean tracks")
    mic_features = causal_acoustic_frames(mic, frame_ms=FRAME_MS).numpy()
    far_features = causal_acoustic_frames(
        tracks["farend_speech"], frame_ms=FRAME_MS
    ).numpy()
    samples_per_frame = SAMPLE_RATE * FRAME_MS // 1000
    padded = np.pad(near, (0, (-len(near)) % samples_per_frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, samples_per_frame) ** 2, axis=1))
    labels = (rms > 10 ** (-50 / 20)).astype(np.int64)
    if len(mic_features) != len(far_features) or len(labels) != len(mic_features):
        raise RuntimeError("AEC feature/label frame alignment failed")
    far_activity = (far_features[:, 0] > 0.25).astype(np.float32)[:, None]
    features = np.concatenate([mic_features, far_features, far_activity], axis=1)
    return {
        "features": features.astype(np.float32),
        "labels": labels,
        "nearend_active_fraction": float(labels.mean()),
        "mixture_residual_rms": residual_rms,
    }


def echo_gain_counterfactual(
    directory: str | Path, fileid: str | int, *, gain: float,
) -> np.ndarray:
    """Re-render mic features with the same near end and a changed echo gain.

    The playback reference is unmodified; the isolated echo waveform is used
    offline only to construct this counterfactual, never as model input.
    """
    if not np.isfinite(gain) or not 0.25 <= gain <= 2.0:
        raise ValueError("counterfactual echo gain must be in [0.25, 2]")
    directory = Path(directory)
    mic, mic_rate = sf.read(directory / TRACK_FILES["nearend_mic"].format(id=fileid),
                            dtype="float32")
    echo, echo_rate = sf.read(directory / TRACK_FILES["echo"].format(id=fileid),
                              dtype="float32")
    far, far_rate = sf.read(directory / TRACK_FILES["farend_speech"].format(id=fileid),
                           dtype="float32")
    if (mic_rate != SAMPLE_RATE or echo_rate != SAMPLE_RATE or far_rate != SAMPLE_RATE
            or mic.ndim != 1 or echo.ndim != 1 or far.ndim != 1
            or not (len(mic) == len(echo) == len(far))):
        raise ValueError("invalid AEC counterfactual input clocks")
    changed_mic = np.clip(mic + (gain - 1.0) * echo, -1.0, 1.0)
    mic_features = causal_acoustic_frames(changed_mic, frame_ms=FRAME_MS).numpy()
    far_features = causal_acoustic_frames(far, frame_ms=FRAME_MS).numpy()
    far_activity = (far_features[:, 0] > 0.25).astype(np.float32)[:, None]
    return np.concatenate([mic_features, far_features, far_activity], axis=1).astype(np.float32)


def echo_delay_counterfactual(
    directory: str | Path, fileid: str | int, *, delay_ms: int,
) -> np.ndarray:
    """Delay only the physical echo; keep user signal and played reference fixed.

    The isolated echo is an offline intervention source, never a model input.
    This probes playback-to-microphone path variation, not network-reference
    timestamp corruption. The new waveform is a controlled synthetic mix.
    """
    if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or not 0 <= delay_ms <= 400:
        raise ValueError("delay_ms must be an integer in [0, 400]")
    directory = Path(directory)
    mic, mic_rate = sf.read(directory / TRACK_FILES["nearend_mic"].format(id=fileid),
                            dtype="float32")
    echo, echo_rate = sf.read(directory / TRACK_FILES["echo"].format(id=fileid),
                              dtype="float32")
    far, far_rate = sf.read(directory / TRACK_FILES["farend_speech"].format(id=fileid),
                            dtype="float32")
    if (mic_rate != SAMPLE_RATE or echo_rate != SAMPLE_RATE or far_rate != SAMPLE_RATE
            or mic.ndim != 1 or echo.ndim != 1 or far.ndim != 1
            or not (len(mic) == len(echo) == len(far))):
        raise ValueError("invalid AEC delay counterfactual input clocks")
    samples = SAMPLE_RATE * delay_ms // 1000
    shifted = np.pad(echo, (samples, 0))[:len(echo)] if samples else echo
    changed_mic = np.clip(mic - echo + shifted, -1.0, 1.0)
    mic_features = causal_acoustic_frames(changed_mic, frame_ms=FRAME_MS).numpy()
    far_features = causal_acoustic_frames(far, frame_ms=FRAME_MS).numpy()
    far_activity = (far_features[:, 0] > 0.25).astype(np.float32)[:, None]
    return np.concatenate([mic_features, far_features, far_activity], axis=1).astype(np.float32)
