"""多方言/多场景前端增强（Direction C）。

把"场景识别 + 方言自适应 + 前端 Adapter 门控"衔接在全双工基座前端之上。
核心：不需要换基座，只在特征进入 LLM 前按场景加权，提高嘈杂/口音下的鲁棒性。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrontContext:
    scened: str = "clean"     # clean|noisy|reverb|farfield
    dialect: str = "mandarin" # mandarin|cantonese|sichuan|english_acc...
    enh_style: str = "light"  # light|medium|aggressive
    gain: float = 1.0


class SceneDialectFrontend:
    """轻量场景/方言自适应前端路由。可替换为更重的 adapter 门控网络。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def classify(self, feat_vector) -> FrontContext:
        """占位：接你已有的场景分类器 / 方言分类器输出。
        可用与 Whisper 场景ASR同源的多分类器结果替换，实现复用。
        """
        ctx = FrontContext()
        return ctx

    def apply(self, feats, ctx: Optional[FrontContext] = None) -> dict:
        """对特征施加场景化增强/增益。占位：真实实现可插音频前处理。"""
        ctx = ctx or self.classify(None)
        mapped = {
            "enh_style": ctx.enh_style,
            "gain": ctx.gain,
            "dialect_bias": ctx.dialect,
        }
        return mapped


def route_gate(front_cfg: dict) -> object:
    """工厂。未来可替换成"多LoRA adapter 门控"以拟合你在 Whisper 上的联合适配经验。"""
    return SceneDialectFrontend(front_cfg)