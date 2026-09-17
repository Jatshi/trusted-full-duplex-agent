"""Barge-in（用户抢话打断）检测与可中断播报。

全双工落地第一刚需：bot 播报中用户插话，系统要在百毫秒级停掉 TTS、
丢弃未播音频，并把"说了一半"的状态正确留给下一轮。

两件事：
  EnergyVAD            麦克风流的用户开口检测（能量门限 + 持续帧确认，防瞬时噪声）
  InterruptiblePlayback 播放侧包装：每消费一个 TTS chunk 前查 VAD，
                       触发即截断——模拟真机"播放线程 vs 麦克风线程"交接。

延迟分解（本地可测的机制部分）：
  detection_latency  用户开口 -> VAD 判定（sustain 帧 × frame_ms + chunk 量化）
  stop_latency       VAD 判定 -> 播放停止（当前 chunk 播完即停，<= chunk_ms）
  total              开口 -> 停止
KV/上下文层面"打断后模型记不记得"需真基座，见远端 63 脚本。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterator
import numpy as np

SR = 16000


@dataclass
class EnergyVAD:
    """能量 VAD：能量连续 sustain_frames 帧过门限即报"用户开口"。"""
    threshold_db: float = -38.0
    frame_ms: int = 100
    sustain_frames: int = 3

    def __post_init__(self):
        self._run = 0
        self.fired = False

    def feed_db(self, db: float) -> bool:
        """喂一帧能量(dB)。返回 True 表示这一帧触发了开口判定（只触发一次）。"""
        if self.fired:
            return False
        if db > self.threshold_db:
            self._run += 1
        else:
            self._run = 0
        if self._run >= self.sustain_frames:
            self.fired = True
            return True
        return False

    def reset(self):
        self._run = 0
        self.fired = False


def frame_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)))) + 1e-9
    return float(np.clip(20.0 * np.log10(rms), -60.0, 0.0))


@dataclass
class PlaybackResult:
    interrupted: bool = False
    played_ms: float = 0.0            # 实际播出的时长
    total_ms: float = 0.0             # 本该播完的时长
    bargein_at_ms: float = None       # 用户开口时刻（麦克风时间轴）
    stopped_at_ms: float = None       # 播放停止时刻（同一时间轴）
    detection_latency_ms: float = None
    stop_latency_ms: float = None
    dropped_chunks: int = 0


class InterruptiblePlayback:
    """逐 chunk 播放 TTS 流，每个 chunk 前查打断。真机上 TTS chunk 来自
    streaming_generate 的迭代器（teacher forcing 澄清播报同理）；
    本类只负责"检查-消费-截断"机制，与音频来源解耦。

    on_interrupt: 截断回调（真机用它停掉音频设备/丢弃输出队列）。
    """

    def __init__(self, vad: EnergyVAD, chunk_ms: int = 200,
                 on_interrupt: Callable[[], None] = None):
        self.vad = vad
        self.chunk_ms = chunk_ms
        self.on_interrupt = on_interrupt

    def play(self, tts_chunks: Iterator[bytes],
             mic_stream: Iterator[np.ndarray],
             frame_ms: int = None) -> PlaybackResult:
        """tts_chunks: bot 音频 chunk 序列；mic_stream: 与播放并行的麦克风帧流。

        时间轴：麦克风监听与 bot 播放同时开始，共用一条时间轴。每个 TTS
        chunk 播放耗时 chunk_ms，期间消费 ceil(chunk_ms/frame_ms) 个麦克风帧
        ——模拟真实"VAD 100ms 一判、TTS 200ms 一块"的节奏。触发即停，
        当前 chunk 的剩余部分与后续 chunk 全部丢弃。
        """
        frame_ms = frame_ms or self.vad.frame_ms
        res = PlaybackResult()
        frames_per_chunk = max(int(round(self.chunk_ms / frame_ms)), 1)
        t_ms = 0.0        # 播放时间轴（已播出毫秒）
        mic_t = 0.0       # 麦克风时间轴（已监听毫秒）
        for _chunk in tts_chunks:
            for _ in range(frames_per_chunk):
                try:
                    mic = next(mic_stream)
                except StopIteration:
                    mic = None
                mic_t += frame_ms
                if mic is not None and self.vad.feed_db(frame_db(mic)):
                    # VAD 在本帧末触发：前 sustain 帧连续过门限，
                    # 用户开口时刻 = 触发时刻回溯 sustain*frame_ms
                    res.interrupted = True
                    res.bargein_at_ms = max(
                        mic_t - self.vad.sustain_frames * frame_ms, 0.0)
                    # 正在播的 chunk 已进音频缓冲无法收回，播到 chunk 边界
                    # 才真正静音；队列中后续 chunk 丢弃。
                    res.stopped_at_ms = float(
                        ((int(mic_t) - 1) // self.chunk_ms + 1) * self.chunk_ms)
                    res.detection_latency_ms = mic_t - res.bargein_at_ms
                    res.stop_latency_ms = res.stopped_at_ms - mic_t
                    res.dropped_chunks = 1
                    if self.on_interrupt:
                        self.on_interrupt()
                    return res
            t_ms += self.chunk_ms
            res.played_ms = t_ms
        res.total_ms = res.played_ms
        return res


def make_mic_stream(bot_len_ms: float, bargein_at_ms: float,
                    bargein_speech_ms: int = 1500, sr: int = SR,
                    frame_ms: int = 100, rng_seed: int = 0) -> Iterator[np.ndarray]:
    """合成麦克风帧流：bargein_at_ms 之前为安静本底，之后为用户插话语音。

    用于本地机制评测：给定"用户何时插话"，测系统何时停播。
    """
    rng = np.random.default_rng(rng_seed)
    fn = int(frame_ms / 1000 * sr)
    total_ms = bot_len_ms + bargein_speech_ms + 500
    t = 0
    while t < total_ms:
        if bargein_at_ms <= t < bargein_at_ms + bargein_speech_ms:
            env = rng.uniform(0.35, 1.0, 8)
            e = np.interp(np.linspace(0, 8, fn), np.arange(8), env)
            frame = (rng.standard_normal(fn) * e).astype(np.float32)
        else:
            frame = (rng.standard_normal(fn) * 0.003).astype(np.float32)
        yield frame
        t += frame_ms
