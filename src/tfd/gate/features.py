"""从流式理解结果中提取：confidence / uncertainty / risk。

设计要点：全双工时延下要"逐 chunk"给出可信信号，
因此所有特征都设计为可增量计算、可缓存。
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class GateFeature:
    confidence: float            # 0~1，对当前意图的理解置信
    risk: float                  # 0~1，动作的高危程度
    entropy: float               # 归一化不确定度 0~1
    intent: Optional[str] = None
    raw_text: Optional[str] = None

    @property
    def decision_score(self) -> float:
        """护栏总分：risk 越高、置信越低 -> 分越高(越不该直接执行)。"""
        return self.risk * (1.0 - self.confidence)


def _normalize_entropy(probs: list[float]) -> float:
    probs = [max(p, 1e-9) for p in probs]
    total = sum(probs)
    probs = [p / total for p in probs]
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(len(probs))
    return (h / h_max) if h_max > 0 else 0.0


# 经验映射：把平均逐 token 负对数似然映射到 0~1 置信
# 例：-0.05 -> 0.95，-1.0 -> 0.4（可在真实数据上重校准）
def _scale_logprob(logprob: Optional[float]) -> float:
    if logprob is None:
        return 0.5
    if logprob >= 0:
        return 1.0
    x = -logprob
    return 1.0 / (1.0 + x)


class ConfidenceEstimator:
    def __init__(self, gate_cfg: dict):
        g = gate_cfg.get("features", {})
        self.source = g.get("confidence_source", "model_logprob")
        self.use_uncertainty = g.get("use_uncertainty", True)

    def estimate(
        self,
        *,
        logprob: Optional[float] = None,
        asr_confidence: Optional[float] = None,
        intent_probs: Optional[list[float]] = None,
    ) -> GateFeature:
        conf = self._confidence(logprob, asr_confidence)
        ent = _normalize_entropy(intent_probs) if intent_probs else 0.5
        if self.use_uncertainty:
            conf = conf * (1.0 - 0.5 * ent)
        return GateFeature(
            confidence=max(0.0, min(1.0, conf)),
            risk=0.0,
            entropy=ent,
        )

    def _confidence(self, logprob, asr_confidence):
        src = self.source
        model_scaled = _scale_logprob(logprob)
        if src == "model_logprob":
            return model_scaled if logprob is not None else 0.5
        if src == "asr_confidence":
            return asr_confidence if asr_confidence is not None else 0.5
        # both_avg
        a = model_scaled if logprob is not None else 0.5
        b = asr_confidence if asr_confidence is not None else 0.5
        return (a + b) / 2.0


class RiskEstimator:
    """基于意图类别 + 高危关键词的流式风险估计。

    不可逆类（删除/转账/支付…）bonus 必须不低于 trust_gate 的
    risk_confirm_floor（0.4）：score 路径 risk*(1-conf) 数学上到不了
    execute_thr，若 bonus 落在 floor 之下，清晰语音的不可逆指令会
    直接 EXECUTE 绕过二次确认（64_gate_incremental 的 clear-voice
    探针复现过该缺陷，故把不可逆类抬到确认线）。
    """

    def __init__(self, gate_cfg: dict):
        r = gate_cfg.get("risk", {})
        self.high_risk = r.get("high_risk_intents", {})
        self.irrev_bonus = r.get("irreversible_bonus", 0.4)
        self.phys_bonus = r.get("physical_bonus", 0.5)

    def estimate(self, text: Optional[str], intent: Optional[str]) -> float:
        risk = 0.0
        irrev = self.high_risk.get("irreversibles", [])
        phys = self.high_risk.get("physical_acts", [])
        if text:
            risk = max(risk, self.irrev_bonus if any(k in text for k in irrev) else risk)
            risk = max(risk, self.phys_bonus if any(k in text for k in phys) else risk)
        return min(1.0, risk)


def combine(feat: GateFeature, risk: float) -> GateFeature:
    feat.risk = risk
    feat.entropy = max(feat.entropy, 0.0)
    return feat