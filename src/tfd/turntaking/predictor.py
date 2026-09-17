"""三态 turn-taking 策略与事件级评测。

三个策略同一接口（逐帧 feed 特征，输出动作），保证对比公平：
  RulePauseStrategy      固定停顿阈值（传统 VAD 方案的代表，500ms 典型值）
  BaseAggressiveStrategy 模拟端到端基座的内置时机策略：停顿够短 + "看起来可答"
                         就开口（对句中停顿激进，容易抢话）。
                         注意：这是策略模拟（可复现），不是基座实测——
                         基座实测见远端 63 脚本。
  LearnedStrategy        训练出的 MLP 决策器（本项目的差异化工作）

评测口径对齐 turn-taking 文献（TurnGPT / VAP）的事件级指标：
  false_takeover  用户没说完就开口（句中停顿触发 take）
  miss            用户说完 1s 内没接话
  take_latency    用户真正说完 -> take 触发 的延迟(ms)
  backchannel     犹豫停顿内发了反馈(precision/recall)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .features import FEATURE_DIM

HOLD, TAKE, BACKCHANNEL = "hold", "take", "backchannel"
ACTIONS = (HOLD, TAKE, BACKCHANNEL)
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ---------------------------------------------------------------- 策略基类
class TurnStrategy:
    name = "abstract"

    def feed(self, feats: list[float]) -> str:
        raise NotImplementedError

    def reset(self):
        pass


class RulePauseStrategy(TurnStrategy):
    """固定停顿阈值：>= threshold_ms 的静音即开口。经典 VAD 后处理方案。"""
    name = "rule_pause"

    def __init__(self, threshold_ms: int = 500):
        self.thr = threshold_ms / 1000.0
        self.reset()

    def feed(self, feats: list[float]) -> str:
        pause_len = feats[1] * 1.5          # 反归一化（与 features.py 一致）
        return TAKE if pause_len >= self.thr else HOLD

    def reset(self):
        pass


class BaseAggressiveStrategy(TurnStrategy):
    """模拟端到端基座内置策略：语义模型倾向于"听到停顿且内容可答就答"，
    停顿阈值低（300ms）且对已说内容量敏感（说得越多越像说完了）。"""
    name = "base_aggressive"

    def __init__(self, threshold_ms: int = 300):
        self.thr = threshold_ms / 1000.0
        self.reset()

    def feed(self, feats: list[float]) -> str:
        pause_len = feats[1] * 1.5
        utt_speech = feats[5] * 6.0
        semantic_ready = min(utt_speech / 1.2, 1.0)     # 累计语音>=1.2s 视为"可答"
        if pause_len >= self.thr and semantic_ready >= 0.8:
            return TAKE
        return HOLD

    def reset(self):
        pass


class LearnedStrategy(TurnStrategy):
    """MLP 决策器：逐帧无状态推理（特征自带时序聚合），车机 CPU 可部署。

    触发规则：帧级概率 + 连续确认（confirm_frames 帧连续预测 take 才开口）。
    原因是代价不对称——句中/结尾停顿在特征空间有重叠区间，单帧预测在该
    区间必然是概率性的；抢话（打断用户）代价远高于晚接 200-300ms，
    用时间确认换错误率，对应 TurnGPT 里 prediction smoothing 的思路。
    """
    name = "learned"

    def __init__(self, hidden: int = 32, weights_path: str = None,
                 confirm_frames: int = 4):
        import torch
        self.torch = torch
        self.confirm_frames = confirm_frames
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(ACTIONS)),
        )
        if weights_path and Path(weights_path).exists():
            self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()
        self.reset()

    def feed(self, feats: list[float]) -> str:
        x = self.torch.tensor([feats], dtype=self.torch.float32)
        with self.torch.no_grad():
            logits = self.net(x)
        pred = ACTIONS[int(logits.argmax(dim=-1).item())]
        if pred == TAKE:
            self._take_run += 1
        else:
            self._take_run = 0
        if self._take_run >= self.confirm_frames:
            return TAKE
        if pred == TAKE:
            return HOLD       # 未攒够确认帧的 take 预测：先按 hold 处理
        return pred

    def reset(self):
        self._take_run = 0

    def probs(self, feats: list[float]) -> list[float]:
        x = self.torch.tensor([feats], dtype=self.torch.float32)
        with self.torch.no_grad():
            p = self.torch.softmax(self.net(x), dim=-1)
        return p.squeeze(0).tolist()


# ---------------------------------------------------------------- 可训练 MLP
class TrainableMLP:
    """训练封装：帧级交叉熵 + 类权重（take 样本少但代价高，加权召回）。"""

    def __init__(self, hidden: int = 32, lr: float = 0.01, seed: int = 42):
        import torch
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(ACTIONS)),
        )
        torch.nn.init.xavier_uniform_(self.net[0].weight, generator=g)
        torch.nn.init.xavier_uniform_(self.net[2].weight, generator=g)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: list[str], epochs: int = 300,
            class_weights: dict = None, verbose_every: int = 100) -> list[float]:
        t = self.torch
        xt = t.tensor(X, dtype=t.float32)
        yt = t.tensor([ACT_IDX[a] for a in y], dtype=t.long)
        w = None
        if class_weights:
            w = t.tensor([class_weights.get(a, 1.0) for a in ACTIONS],
                         dtype=t.float32)
        losses = []
        for ep in range(epochs):
            self.opt.zero_grad()
            logits = self.net(xt)
            loss = t.nn.functional.cross_entropy(logits, yt, weight=w)
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
            if verbose_every and (ep + 1) % verbose_every == 0:
                print(f"    epoch {ep+1}: loss={loss.item():.4f}")
        return losses

    def accuracy(self, X: np.ndarray, y: list[str]) -> float:
        t = self.torch
        with t.no_grad():
            pred = self.net(t.tensor(X, dtype=t.float32)).argmax(-1).tolist()
        gold = [ACT_IDX[a] for a in y]
        return float(np.mean([p == g for p, g in zip(pred, gold)]))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.net.state_dict(), path)

    def to_strategy(self, confirm_frames: int = 3) -> LearnedStrategy:
        """confirm_frames=1 即纯帧级分类（无确认平滑），供帧级统计用。"""
        s = LearnedStrategy(confirm_frames=confirm_frames)
        s.net.load_state_dict(self.net.state_dict())
        return s


# ---------------------------------------------------------------- 事件级评测
@dataclass
class TurnEvent:
    """一条话语上的策略行为事件记录。"""
    false_takeover: bool = False
    miss: bool = False
    take_latency_ms: float = None
    backchannel_fired_in_filler: bool = False
    backchannel_fired_outside: bool = False
    take_at_ms: float = None


def evaluate_strategy(strategy: TurnStrategy, feats_seq: np.ndarray,
                      labels: list[str], frame_ms: int = 100,
                      miss_latency_ms: int = 1000,
                      final_pause_start_ms: float = None) -> TurnEvent:
    """流式跑一遍策略（每帧 feed，首次 take 即开口），按事件口径打分。

    - take 触发于 final_pause 开始之前 -> false_takeover（抢话）
    - final_pause 开始后 miss_latency 内触发 -> 正常，记延迟
    - 超时或从未触发 -> miss（漏接）
    - backchannel 在 filler 标签帧触发为正确，其余帧触发为噪声
    """
    assert final_pause_start_ms is not None, "ground truth 时刻必须显式传入（勿从标签反推）"
    final_start_ms = final_pause_start_ms
    take_frames = [i for i, a in enumerate(labels) if a == TAKE]

    ev = TurnEvent()
    strategy.reset()
    for i, feats in enumerate(feats_seq):
        act = strategy.feed(list(feats))
        t_ms = (i + 1) * frame_ms
        if act == TAKE:
            ev.take_at_ms = t_ms
            ev.false_takeover = t_ms <= final_start_ms
            if not ev.false_takeover:
                ev.take_latency_ms = t_ms - final_start_ms
                ev.miss = ev.take_latency_ms > miss_latency_ms
            break
        if act == BACKCHANNEL:
            if labels[i] == BACKCHANNEL:
                ev.backchannel_fired_in_filler = True
            else:
                ev.backchannel_fired_outside = True
    if ev.take_at_ms is None:
        # 从未 take：若话语有 take 标签则按超时记 miss
        if take_frames:
            ev.miss = True
            ev.take_latency_ms = float("nan")
    return ev
