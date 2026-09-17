"""合成流式决策点数据：用户话语的"语音段-停顿段"结构 + 帧级三态标签。

对话规律的参数化（分布有刻意重叠，单一停顿阈值无法完美分类——
这正是联合时序特征的价值所在）：
  句中停顿 pause        U(0.15, 0.45)s  用户还会继续说 -> hold
  犹豫停顿 filler_pause U(0.35, 0.65)s  填充词停顿，思考中 -> backchannel 机会
  结尾停顿 final_pause  U(0.70, 1.20)s  说完了 -> take
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

HOLD, TAKE, BACKCHANNEL = "hold", "take", "backchannel"
SR = 16000


@dataclass
class Utterance:
    """一条用户话语：段序列 + 元信息。音频由段结构确定性合成。"""
    segs: list                       # [("speech"|"pause"|"filler_pause"|"final_pause", dur_s), ...]
    sr: int = SR
    rng_state: int = 0               # 音频合成用的独立种子（段结构由外层种子决定）
    _audio_cache: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def total_s(self) -> float:
        return sum(d for _, d in self.segs)

    @property
    def final_pause_start(self) -> float:
        """结尾停顿的起点（= 用户真正说完的时刻，评测的 ground truth）。"""
        t = 0.0
        for kind, d in self.segs:
            if kind == "final_pause":
                return t
            t += d
        return t

    @property
    def filler_windows(self) -> list[tuple[float, float]]:
        """犹豫停顿窗口 [start, end)。"""
        out, t = [], 0.0
        for kind, d in self.segs:
            if kind == "filler_pause":
                out.append((t, t + d))
            t += d
        return out

    def to_audio(self) -> np.ndarray:
        """语音段=幅度调制的浊音样噪声，停顿段=低电平本底噪声。

        声学规律编码（否则句中/结尾停顿的特征不可分，模型只能靠停顿时长赌）：
        - 结尾停顿前紧邻的语音段末尾叠加能量衰减包络（1.0->0.3，模拟
          说完时的语调下降/能量收尾），energy_slope 因此有判别力
        - 句中/犹豫停顿前的语音段保持平坦包络（话说一半，无收尾征兆）
        """
        if self._audio_cache is not None:
            return self._audio_cache
        rng = np.random.default_rng(self.rng_state)
        # 找 final_pause 前紧邻的 speech 段下标（衰减目标）
        final_prev_speech = None
        for i, (kind, _d) in enumerate(self.segs):
            if kind == "final_pause":
                j = i - 1
                if j >= 0 and self.segs[j][0] == "speech":
                    final_prev_speech = j
                break
        parts = []
        for idx, (kind, dur) in enumerate(self.segs):
            n = int(round(dur * self.sr))
            if kind == "speech":
                # 平滑包络：2 个低频正弦叠加（音节级能量起伏）。
                # 不用独立采样控制点——那会产生陡峭跳变，污染 pre_pause_drop 特征
                t = np.linspace(0, 1, n)
                f1, f2 = rng.uniform(0.5, 1.5), rng.uniform(1.0, 2.0)
                p1, p2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
                env = (0.65 + 0.20 * np.sin(2 * np.pi * f1 * t + p1)
                       + 0.12 * np.sin(2 * np.pi * f2 * t + p2))
                env = np.clip(env, 0.30, 1.0).astype(np.float32)
                if idx == final_prev_speech and n > 0:
                    tail = max(int(n * 0.4), 1)          # 末 40% 叠加衰减
                    decay = np.linspace(1.0, 0.12, tail)
                    env[-tail:] = env[-tail:] * decay
                parts.append((rng.standard_normal(n) * env).astype(np.float32))
            else:
                parts.append((rng.standard_normal(n) * 0.003).astype(np.float32))
        self._audio_cache = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return self._audio_cache


def _gen_utterance(rng: np.random.Generator) -> Utterance:
    n_speech = int(rng.integers(1, 6))            # 1~5 个语音段
    has_filler = rng.random() < 0.35              # 35% 话语含犹豫停顿
    segs = [("speech", float(rng.uniform(0.30, 0.90)))]
    filler_done = not has_filler
    for i in range(n_speech - 1):
        if not filler_done and i == 0 and rng.random() < 0.7:
            # 犹豫停顿倾向出现在话语前半段（用户边想边说）
            segs.append(("filler_pause", float(rng.uniform(0.35, 0.65))))
            filler_done = True
        else:
            segs.append(("pause", float(rng.uniform(0.15, 0.45))))
        segs.append(("speech", float(rng.uniform(0.30, 0.90))))
    if not filler_done:
        segs.append(("filler_pause", float(rng.uniform(0.35, 0.65))))
    segs.append(("final_pause", float(rng.uniform(0.70, 1.20))))
    # 真实对话规律编码（特征可分性的来源，均为可观测信号）：
    #   犹豫停顿多跟在短填充词后（"嗯…"前的短语），结尾停顿跟在
    #   较长的完整句后——pre_pause_speech_s 因此区分 filler / final
    final_i = len(segs) - 1
    if final_i >= 1 and segs[final_i - 1][0] == "speech":
        segs[final_i - 1] = ("speech", float(rng.uniform(0.60, 1.20)))
    for i in range(1, len(segs)):
        if (segs[i][0] == "filler_pause" and segs[i - 1][0] == "speech"
                and i - 1 != final_i - 1):
            segs[i - 1] = ("speech", float(rng.uniform(0.20, 0.45)))
    return Utterance(segs=segs, rng_state=int(rng.integers(0, 2**31)))


def frame_labels(utt: Utterance, frame_ms: int = 100, label_delay_s: float = 0.2) -> list[str]:
    """帧级标签。label_delay_s：停顿进入该时长后才给出 take/backchannel 标签——
    更早的帧连"这是一个可行动停顿"的信号都不完整，强行标注只会教模型猜。"""
    n_frames = int(np.ceil(utt.total_s / (frame_ms / 1000)))
    labels = [HOLD] * n_frames
    fs = frame_ms / 1000
    # take：结尾停顿 start+delay 之后的所有帧
    t0 = utt.final_pause_start + label_delay_s
    for i in range(n_frames):
        if (i + 1) * fs > t0:
            labels[i] = TAKE
    # backchannel：犹豫窗口 start+delay 之后、end 之前的帧（take 不存在冲突：窗口互斥）
    for s, e in utt.filler_windows:
        for i in range(n_frames):
            tc = (i + 1) * fs
            if s + label_delay_s < tc <= e:
                labels[i] = BACKCHANNEL
    return labels


def synth_dataset(n: int, seed: int = 42) -> list[Utterance]:
    rng = np.random.default_rng(seed)
    return [_gen_utterance(rng) for _ in range(n)]
