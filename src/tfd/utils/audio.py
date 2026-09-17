"""音频编解码与小工据工具（与后端约定：输入 pcm16/16k，输出 pcm16/24k）。"""
from __future__ import annotations
import io
import wave
from pathlib import Path

import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return arr / 32768.0


def float_to_pcm16(vals) -> bytes:
    if hasattr(vals, "detach"):
        vals = vals.detach().cpu().numpy()
    arr = np.asarray(vals).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0) * 32767.0
    return arr.astype("<i2").tobytes()


def chunks(pcm16: bytes, chunk_bytes: int) -> list[bytes]:
    """按字节把 pcm16 切成等长块（尾部不足补零到 chunk_bytes）。"""
    if not pcm16:
        return []
    out = []
    for i in range(0, len(pcm16), chunk_bytes):
        seg = pcm16[i : i + chunk_bytes]
        if len(seg) < chunk_bytes:
            seg = seg + b"\x00" * (chunk_bytes - len(seg))
        out.append(seg)
    return out


def write_wav(path, pcm16: bytes, sample_rate: int = 24000):
    """把 pcm16 字节写成单声道 WAV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)


def read_wav_pcm16(path, target_sr: int = 16000) -> bytes:
    """读任意 WAV 并重采样到 target_sr，返回 pcm16 字节（静音兜底）。"""
    import soundfile as sf
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        data = np.zeros(target_sr, dtype=np.float32)
        sr = target_sr
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return float_to_pcm16(data.astype(np.float32))


def _resample(data, src_sr, dst_sr):
    n = int(round(len(data) * dst_sr / src_sr))
    idx = np.linspace(0, len(data) - 1, n)
    return np.interp(idx, np.arange(len(data)), data).astype(np.float32)


def make_test_signal(seconds: float = 1.6, sample_rate: int = 16000,
                     freq: float = 220.0) -> bytes:
    """无真实录音时生成可让模型 `有音频输入` 的测试信号（占位，非真实人声）。

    真实演示请在 data/user_turns/<name>.wav 放置 16k 录音，脚本会优先用它。
    """
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.35 * np.sin(2 * np.pi * freq * t) * _fade(t, seconds)
    return float_to_pcm16(tone.astype(np.float32))


def _fade(t, seconds):
    f = np.ones_like(t)
    n = int(0.1 * len(t))
    if n > 0:
        f[:n] *= np.linspace(0, 1, n)
        f[-n:] *= np.linspace(1, 0, n)
    return f