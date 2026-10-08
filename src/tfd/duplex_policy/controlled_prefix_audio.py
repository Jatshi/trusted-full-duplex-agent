"""Fixed E72 replay input transform; EOF metadata never enters model features.

No file decoding, transcript access, silence selection, or model execution.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly


def prepare_prefix(audio: np.ndarray, *, source_rate: int) -> np.ndarray:
    """Take exactly the first 30 s of mono 48 kHz and resample to 16 kHz."""
    audio = np.asarray(audio)
    if source_rate != 48000 or audio.ndim != 1 or audio.size < 1440000:
        raise ValueError("Expected mono 48 kHz with at least 1440000 samples")
    prefix = np.asarray(audio[:1440000], dtype=np.float32)
    if not np.isfinite(prefix).all():
        raise ValueError("Nonfinite prefix samples")
    result = resample_poly(prefix, 1, 3, window=("kaiser", 5.0), padtype="constant")
    if result.size != 480000 or not np.isfinite(result).all():
        raise ValueError("Invalid resampled prefix")
    return result.astype(np.float32, copy=False)


@dataclass(frozen=True)
class ReplayPacket:
    audio: np.ndarray
    start_sample: int
    real_samples: int
    artificial_samples: int
    eof_reached: bool


def replay_packets(prefix: np.ndarray) -> Iterator[ReplayPacket]:
    """Append 15 s artificial silence, without padding or dropping last packet.

    eof_reached describes the packet END reaching the finite source boundary.
    It is measurement metadata, not turn completion or a policy input.
    """
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.ndim != 1 or prefix.size != 480000 or not np.isfinite(prefix).all():
        raise ValueError("Expected finite mono 480000-sample prefix")
    replay = np.concatenate((prefix, np.zeros(240000, dtype=np.float32)))
    cursor = 0
    while cursor < replay.size:
        end = min(cursor + (16560 if cursor == 0 else 16000), replay.size)
        real = max(0, min(end, prefix.size) - min(cursor, prefix.size))
        yield ReplayPacket(replay[cursor:end].copy(), cursor, real, end - cursor - real, end >= prefix.size)
        cursor = end
