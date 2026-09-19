"""帧特征提取：全部设计为流式可增量计算（无未来信息，不泄漏标签）。

9 维特征（无状态逐帧推理即可用——部署到车机端 CPU 不需要保留 RNN 隐状态）：
  energy              当前帧 RMS（dB 相对满幅）
  pause_len           当前连续静音时长(s)
  energy_slope        最近 0.5s 能量趋势(线性斜率，负=衰减，说完先兆)
  speech_ratio_1s     最近 1s 语音活跃占比
  speech_ratio_3s     最近 3s 语音活跃占比（长时语速代理）
  utt_speech_s        本话语累计语音时长
  n_inner_pauses      已出现的句中/犹豫停顿次数
  pre_pause_speech_s  最近一次停顿前那段语音的时长（短语音+长停顿更像犹豫）
  last_voiced_energy  最近一个语音帧的能量(dB 归一化)。说完的语调衰减
                      （停顿前能量已收低）与话说一半的急停（停顿前能量
                      仍高）由此可分——turn-taking 的关键声学先兆。
  pre_pause_drop      停顿前语音尾部的能量下降量(dB)：最近 voiced 帧相对
                      其 ~400ms 前 voiced 帧的降幅。逐渐收尾(说完)有
                      明显下降；高能量急停(话说一半)降幅≈0。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .synth import Utterance, SR

FEATURE_DIM = 10
FEATURE_NAMES = [
    "energy_db", "pause_len", "energy_slope", "speech_ratio_1s",
    "speech_ratio_3s", "utt_speech_s", "n_inner_pauses", "pre_pause_speech_s",
    "last_voiced_energy", "pre_pause_drop",
]

ENERGY_FLOOR_DB = -60.0


def frame_energy_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)))) + 1e-9
    db = 20.0 * np.log10(rms)
    return float(np.clip(db, ENERGY_FLOOR_DB, 0.0))


@dataclass
class FrameFeaturizer:
    """流式特征器：feed 逐帧推进，state 逐步累积。"""
    frame_ms: int = 100
    vad_db: float = -38.0             # 语音/静音分界（合成语音~[-15,-3]dB，噪底~-50dB）
    slope_win_s: float = 0.5

    def __post_init__(self):
        self.fs = self.frame_ms / 1000
        self.frame_n = int(round(self.frame_ms / 1000 * SR))
        self.slope_n = max(int(self.slope_win_s / self.fs), 2)
        self._hist: list[float] = []          # 每帧能量 dB 历史
        self._pause_len = 0.0
        self._utt_speech = 0.0
        self._n_pauses = 0
        self._pre_pause_speech = 0.0
        self._cur_speech_run = 0.0            # 当前（最近一段）连续语音时长
        self._last_voiced_db = ENERGY_FLOOR_DB
        self._voiced_hist: list[float] = []   # voiced 帧能量历史（停顿时冻结）
        self._seg_start = 0                   # 最近语音段在 voiced_hist 中的起点

    def feed(self, frame: np.ndarray) -> list[float]:
        e = frame_energy_db(frame)
        voiced = e > self.vad_db
        if voiced:
            if self._cur_speech_run == 0:     # 新语音段开始
                self._seg_start = len(self._voiced_hist)
            self._pause_len = 0.0
            self._utt_speech += self.fs
            self._cur_speech_run += self.fs
            self._last_voiced_db = e
            self._voiced_hist.append(e)
        else:
            if self._cur_speech_run > 0:
                self._pre_pause_speech = self._cur_speech_run
                self._cur_speech_run = 0.0
                self._n_pauses += 1
            self._pause_len += self.fs
        self._hist.append(e)

        slope = self._slope()
        r1 = self._speech_ratio(1.0)
        r3 = self._speech_ratio(3.0)
        # 段内降幅：只比较最近语音段内部的尾部帧，跨段比较无意义
        # （上一段尾部高能量 vs 新段低谷相位会产生假降幅）
        drop = 0.0
        lo = max(self._seg_start, len(self._voiced_hist) - 4)
        if len(self._voiced_hist) - lo >= 2:
            drop = max(0.0, self._voiced_hist[lo] - self._voiced_hist[-1])
        # 归一化到 ~[0,1] 量级，MLP 训练稳定
        return [
            (e - ENERGY_FLOOR_DB) / -ENERGY_FLOOR_DB,
            min(self._pause_len / 1.5, 1.0),
            float(np.clip(slope / 40.0, -1.0, 1.0)),
            r1, r3,
            min(self._utt_speech / 6.0, 1.0),
            min(self._n_pauses / 6.0, 1.0),
            min(self._pre_pause_speech / 1.5, 1.0),
            (self._last_voiced_db - ENERGY_FLOOR_DB) / -ENERGY_FLOOR_DB,
            min(drop / 25.0, 1.0),
        ]

    def _slope(self) -> float:
        if len(self._hist) < 2:
            return 0.0
        w = np.asarray(self._hist[-self.slope_n:], dtype=np.float64)
        if len(w) < 2:
            return 0.0
        # 一元最小二乘的闭式斜率。这里窗口最多只有约 5 帧，不需要调用
        # np.polyfit -> LAPACK lstsq；闭式计算更快，也规避部分 Windows MKL
        # 组合在高频小矩阵 lstsq 上出现的进程级 abort。
        x = np.arange(len(w), dtype=np.float64)
        x -= x.mean()
        denom = float(np.dot(x, x))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(x, w - w.mean()) / denom)   # dB/帧

    def _speech_ratio(self, win_s: float) -> float:
        n = int(round(win_s / self.fs))
        w = self._hist[-n:] if n > 0 else []
        if not w:
            return 0.0
        return sum(1 for e in w if e > self.vad_db) / len(w)


def featurize_utterance(utt: Utterance, frame_ms: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """整条话语离线帧化（训练用）。返回 (features[T,8], frame_dbs[T])。"""
    fz = FrameFeaturizer(frame_ms=frame_ms)
    audio = utt.to_audio()
    fn = int(round(frame_ms / 1000 * utt.sr))
    feats, dbs = [], []
    for i in range(0, len(audio), fn):
        frame = audio[i:i + fn]
        if frame.size < fn:
            frame = np.pad(frame, (0, fn - frame.size))
        feats.append(fz.feed(frame))
        dbs.append(frame_energy_db(frame))
    return np.asarray(feats, np.float32), np.asarray(dbs, np.float32)
