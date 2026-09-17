"""turn-taking RL 的 reward 设计。

核心判断：全双工对齐的关键不在算法而在"reward 怎么定义、怎么才不骗模型"。
这里实现五类 reward，全部归一化到 0~1，再加权合成，便于在 GRPO 里做组内相对优势。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnSample:
    """一条对话轮转样本（供 reward 打分）。"""

    turn_timing_score: float = 0.0      # 开口/闭嘴时机合理性 0~1
    barge_in_grace: float = 0.0         # 被打断后顺滑度 0~1
    content_bleu: float = 0.0           # 内容一致性(与上文) 0~1 -> 0.3 时触发惩罚
    naturalness: float = 0.0            # 语速/停顿接近人 0~1
    gate_align: float = 0.0             # 与护栏一致：该停/该澄清是否照做
    meta: dict = field(default_factory=dict)


def turn_timing(t_ms: float, ideal_ms: float = 350.0) -> float:
    """开口时机 reward：越接近理想值越高。"""
    d = abs(t_ms - ideal_ms) / 500.0
    return max(0.0, 1.0 - d)


def barge_in_grace(regroup_ms: float, budget_ms: float = 1200.0) -> float:
    """被打断后重组上线的速度 reward。"""
    return max(0.0, 1.0 - regroup_ms / budget_ms)


def content_consistency(bleu: float) -> float:
    """内容一致性 reward；过低(=信息被洗掉)强烈惩罚。"""
    if bleu is None or bleu < 0.3:
        return 0.0
    return min(1.0, bleu / 0.9)


def naturalness(pause_jitter: float, ok_jitter: float = 0.5) -> float:
    """停顿分布贴近人 reward。jitter 过大(=机械)给惩罚。"""
    return max(0.0, 1.0 - pause_jitter / ok_jitter)


def gate_align(gate_expected: str, model_action: str) -> float:
    """与可信护栏是否一致。模型"该停却直接执行"给 0。"""
    return 1.0 if model_action == gate_expected else 0.0


def total_reward(s: TurnSample, weights: dict) -> float:
    w = {
        "turn_timing": 1.0,
        "barge_in_grace": 1.0,
        "content_consistency": 1.0,
        "naturalness": 0.5,
        "safety_gate_align": 0.7,
    }
    w.update(weights or {})
    return (
        w["turn_timing"] * turn_timing(s.turn_timing_score)
        + w["barge_in_grace"] * barge_in_grace(s.barge_in_grace)
        + w["content_consistency"] * content_consistency(s.content_bleu)
        + w["naturalness"] * naturalness(s.naturalness)
        + w["safety_gate_align"] * gate_align(
            s.meta.get("gate_expected", "execute"), s.meta.get("model_action", "execute")
        )
    )


def grpo_advantage(rewards: list[float]) -> list[float]:
    """GRPO 组内相对优势（z-score）——免 critic 的核心。"""
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var + 1e-8)
    return [(r - mean) / std for r in rewards]