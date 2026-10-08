"""全双工可量化指标：首 token 延迟、打断延迟、误执行率、并听并说占比、内容一致性。

这些指标专门设计成"能写进简历的一句话数字"，也用于护栏消融对比。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    first_token_latency_s: Optional[float] = None
    barge_in_latency_s: Optional[float] = None
    misuse_rate: Optional[float] = None
    takeover_latency_s: Optional[float] = None
    content_bleu: Optional[float] = None
    duplex_simultaneity: Optional[float] = None
    turn_timing_score: Optional[float] = None
    n_scenarios: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in {
                "first_token_latency_s": self.first_token_latency_s,
                "barge_in_latency_s": self.barge_in_latency_s,
                "misuse_rate": self.misuse_rate,
                "takeover_latency_s": self.takeover_latency_s,
                "content_bleu": self.content_bleu,
                "duplex_simultaneity": self.duplex_simultaneity,
                "turn_timing_score": self.turn_timing_score,
            }.items()
            if v is not None
        }


def first_token_latency(input_push_ts: float, first_token_ts: float) -> float:
    """用户最后一个关键音素 -> bot 第一个 token 的时间差。"""
    return max(0.0, first_token_ts - input_push_ts)


def barge_in_latency(barge_in_ts: float, bot_pause_ts: float) -> float:
    """用户插入 -> bot 开始停下的时间差。越小越自然。"""
    return max(0.0, bot_pause_ts - barge_in_ts)


def misuse_rate(misclassified: int, total: int) -> float:
    """护栏把"应澄清/应停"误判成"执行"的比例（越低越好）。"""
    return (misclassified / total) if total > 0 else 0.0


def duplex_simultaneity(overlap_seconds: float, total_seconds: float) -> float:
    """会话中"并听并说"的时长占比。适合 = 全双工真正在跑。"""
    return min(1.0, overlap_seconds / max(total_seconds, 1e-9))


def content_bleu(reference: str, hypothesis: str) -> float:
    """内容一致性（BLEU-1 简化近似）。判断增强/对话是否洗掉关键信息。"""
    if not hypothesis:
        return 0.0
    ref_tokens = _tok(reference)
    hyp_tokens = _tok(hypothesis)
    if not hyp_tokens:
        return 0.0
    hit = sum(t in ref_tokens for t in hyp_tokens)
    precision = hit / len(hyp_tokens)
    brevity = len(hyp_tokens) / max(len(ref_tokens), 1)
    brevity_penalty = min(1.0, brevity) if len(hyp_tokens) < len(ref_tokens) else 1.0
    return precision * brevity_penalty


def bigram_overlap(a: str, b: str) -> float:
    """字符 bigram Jaccard：两段中文文本的内容重叠度。

    63 的探针上下文保持指标与 barge-in context RL 的 context_recall reward
    共用同一口径，保证"训练优化什么"与"评测测什么"一致。
    """
    def bg(s):
        s = "".join(ch for ch in s if ch.isalnum())
        return {s[i:i + 2] for i in range(len(s) - 1)}
    A, B = bg(a), bg(b)
    return len(A & B) / max(len(A | B), 1)


def _tok(s: str):
    # 汉字按字，英文按词；够用于指标演示
    import re

    words = re.findall(r"[a-zA-Z0-9']+", s.lower())
    han = re.findall(r"[\u4e00-\u9fff]", s)
    return words + han
