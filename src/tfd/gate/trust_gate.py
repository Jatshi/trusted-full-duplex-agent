"""流式可信护栏：三态门控（execute / clarify / stop）。

复用自你在 EvidenceAgent/StreamSense/SafeVLA 中的"该不该动"决策思想，
把它接在全双工时延环境下：对每个流式 chunk 增量更新决策，
并用"连续多数确认"防止抖动（config gate.yaml 的 confirm_ratio）。
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from collections import deque

from .features import (
    AmbiguityEstimator,
    ConfidenceEstimator,
    RiskEstimator,
    GateFeature,
    combine,
)

logger = logging.getLogger("tfd.gate")

EXECUTE = "execute"
CLARIFY = "clarify"
STOP = "stop"
DECISIONS = (EXECUTE, CLARIFY, STOP)


@dataclass
class StreamDecision:
    action: str = CLARIFY            # 兜底：不确定时宁可澄清
    score: float = 0.0
    confidence: float = 0.0
    risk: float = 0.0
    rationale: str = ""
    confirmed: bool = False
    history: list = field(default_factory=list)   # 最近 chunk 的决策序列


class TrustGate:
    def __init__(self, gate_cfg: dict):
        t = gate_cfg["thresholds"]
        self.execute_thr = t["execute_thr"]
        self.clarify_thr = t["clarify_thr"]
        self.min_conf = t["min_confidence"]
        s = gate_cfg["streaming"]
        self.stream_mode = s.get("stream_mode", True)
        self.confirm_required = s.get("confirm_required", False)
        self.confirm_ratio = s.get("confirm_ratio", 0.7)
        self.window = int(self.confirm_ratio * 10) + 2  # 连续多数窗口
        self.conf_est = ConfidenceEstimator(gate_cfg)
        self.risk_est = RiskEstimator(gate_cfg)
        self.ambiguity_est = AmbiguityEstimator(gate_cfg)
        # 风险达此阈值绝不直接执行（避免"听清了就乱动"）
        self.risk_floor = gate_cfg.get("risk", {}).get("risk_confirm_floor", 0.4)
        self._decisions: deque = deque(maxlen=self.window)
        # 风险锁存：首个风险帧一旦出现即置位（只升不降），供上层在用户
        # 还没说完时就预取澄清话术/挂起执行路径（chunk 级增量决策的预警信号）
        self.risk_seen = False
        self.first_risk_frame = None
        self._frame_idx = 0

    # ---- 对单个 chunk 的增量评估 ----
    def on_frame(
        self,
        *,
        text: str = "",
        intent: str = None,
        intent_probs: list[float] = None,
        logprob: float = None,
        asr_confidence: float = None,
    ) -> StreamDecision:
        feat = self.conf_est.estimate(
            logprob=logprob, asr_confidence=asr_confidence, intent_probs=intent_probs
        )
        feat.raw_text = text
        feat.ambiguous = self.ambiguity_est.estimate(text)
        risk = self.risk_est.estimate(text, intent)
        if risk > 0 and not self.risk_seen:
            self.risk_seen = True
            self.first_risk_frame = self._frame_idx
        self._frame_idx += 1
        feat = combine(feat, risk)

        action, score, reason = self._decide(feat)
        self._decisions.append(action)

        confirmed = self._is_confirmed(action) if self.confirm_required else (
            True if score >= self.clarify_thr else False
        )
        return StreamDecision(
            action=action,
            score=score,
            confidence=feat.confidence,
            risk=feat.risk,
            rationale=reason,
            confirmed=confirmed,
            history=list(self._decisions),
        )

    # ---- 核心三态映射 ----
    def _decide(self, feat: GateFeature) -> tuple[str, float, str]:
        s = feat.decision_score
        conf = feat.confidence
        # ASR can be confident about a linguistically vague request.  This is
        # semantic uncertainty, not acoustic uncertainty, so it needs its own
        # explicit path instead of abusing the ASR-confidence threshold.
        if feat.ambiguous:
            return CLARIFY, s, "underspecified target -> clarify"
        # 绝对置信度兜底：完全听不出时即使风险低也强制澄清；
        # 若叠加了风险，则宁停不猜（不做任何不安全动作）
        if conf < self.min_conf:
            if feat.risk > 0:
                return STOP, s, f"low confidence + high risk -> stop (conf {conf:.2f})"
            return CLARIFY, s, f"confidence {conf:.2f} < min {self.min_conf}"
        # 高风险动作绝不直接执行：哪怕听清了也要二次确认（甚至停止）
        if feat.risk >= self.risk_floor:
            action = STOP if feat.risk >= 2 * self.risk_floor else CLARIFY
            tag = "STOP" if action == STOP else "confirm"
            return action, s, f"high-risk(risk {feat.risk:.2f}) -> {tag}, no auto-execute"
        if s >= self.clarify_thr:
            return STOP, s, f"high-risk+low-conf -> stop (score {s:.2f})"
        if s >= self.execute_thr:
            return CLARIFY, s, f"medium score {s:.2f} -> clarify first"
        return EXECUTE, s, f"low score {s:.2f} -> execute"

    # ---- 连续多数确认（防抖动）----
    def _is_confirmed(self, action: str) -> bool:
        if not self._decisions:
            return False
        n = len(self._decisions)
        same = sum(1 for a in self._decisions if a == action)
        return (same / n) >= self.confirm_ratio

    def finalize(self) -> StreamDecision:
        """整句结束时的最终决策（从确认结果里取稳定多数）。"""
        if not self._decisions:
            return StreamDecision(action=CLARIFY, rationale="empty stream")
        best = max(DECISIONS, key=lambda a: sum(x == a for x in self._decisions))
        return StreamDecision(
            action=best,
            confidence=self._last_conf(EXECUTE),
            risk=0.0,
            rationale=f"final majority over {len(self._decisions)} frames",
            confirmed=True,
            history=list(self._decisions),
        )

    def _last_conf(self, _a) -> float:
        return 0.5

    def reset(self):
        self._decisions.clear()
        self.risk_seen = False
        self.first_risk_frame = None
        self._frame_idx = 0
