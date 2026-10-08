"""Causal playback-lag feature bank for an AEC source-attribution baseline.

The bank uses only the microphone frame and playback frames at or before it.
It deliberately does not consume the isolated echo or clean near-end tracks.
"""
from __future__ import annotations

import numpy as np


LAGS_FRAMES = (0, 2, 4, 8)  # 0/80/160/320 ms at a 40-ms frame clock.


def causal_lag_bank(features: np.ndarray, *, lags: tuple[int, ...] = LAGS_FRAMES) -> np.ndarray:
    """Return [clips, frames, 21 + 12*lags] without using future frames.

    Each lag contributes the 10 playback features, a microphone/playback
    energy difference, and a validity bit so padding cannot masquerade as
    genuine playback silence.
    """
    data = np.asarray(features, dtype=np.float32)
    if (data.ndim != 3 or data.shape[-1] != 21 or not np.isfinite(data).all()
            or not lags or any(isinstance(lag, bool) or not isinstance(lag, int)
                                or lag < 0 for lag in lags)
            or len(set(lags)) != len(lags)):
        raise ValueError("expected finite [clip,frame,21] features and unique causal lags")
    clips, frames, _ = data.shape
    pieces = [data]
    for lag in lags:
        delayed = np.zeros((clips, frames, 10), dtype=np.float32)
        valid = np.zeros((clips, frames, 1), dtype=np.float32)
        if lag < frames:
            delayed[:, lag:] = data[:, :frames - lag, 10:20]
            valid[:, lag:] = 1.0
        energy_gap = data[:, :, 0:1] - delayed[:, :, 0:1]
        pieces.extend((delayed, energy_gap, valid))
    return np.concatenate(pieces, axis=-1)
