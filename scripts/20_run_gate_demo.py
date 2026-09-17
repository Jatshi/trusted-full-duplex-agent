#!/usr/bin/env python3
"""20 · 流式可信护栏 demo（核心差异化演示）。

--offline 时用合成 logprob/asr_confidence 输入跑三态门控，无需 GPU、任何机器可跑，
用于在 AutoDL 上机前先看到护栏逻辑生效；去掉 --offline 则接真实基座(见 base/backends)。
"""
from __future__ import annotations
import argparse

from bootstrap import load_config
from tfd.gate.trust_gate import TrustGate, EXECUTE, CLARIFY, STOP


SCENARIOS = [
    # (name, text, risk hint) —— 用 text 里的高危关键词触发 risk
    ("ambiguous", "帮我开一下那个", {}),
    ("high_risk_reversible", "帮我把明早七点的闹钟设一下", {}),
    ("safety_critical", "现在就右转然后变道", {}),
    ("low_risk_clear", "今天天气怎么样", {}),
    ("delete_phrased", "帮我把相册里的视频删了", {}),
]


def run_offline(gate_cfg) -> int:
    print("=" * 72)
    print("流式可信护栏 demo (离线合成输入，无 GPU 也可跑)")
    print("=" * 72)
    gate = TrustGate(gate_cfg)
    low_conf = gate_cfg["thresholds"]["min_confidence"]
    for name, text, _extra in SCENARIOS:
        print(f"\n--- 场景: {name} | 文本: 「{text}」 ---")
        # 模拟逐 chunk：前两帧置信低，后程稳定；并给出 logprob
        decision = None
        for k in range(4):
            conf_logprob = (-0.08 if k >= 2 else -0.6)   # 后程置信更高
            decision = gate.on_frame(
                text=text,
                intent=None,
                logprob=conf_logprob,
                asr_confidence=0.5 + 0.2 * (k >= 2),
            )
            pair = {EXECUTE: "执行", CLARIFY: "澄清", STOP: "停止"}[decision.action]
            print(
                f"  chunk{k}: {pair:<4} score={decision.score:.2f} "
                f"conf={decision.confidence:.2f} risk={decision.risk:.2f}"
            )
        final = gate.finalize()
        pair = {EXECUTE: "执行", CLARIFY: "澄清", STOP: "停止"}[final.action]
        print(f"  最终决策: {pair} ({final.rationale})")
        gate.reset()
    print("\n" + "=" * 72)
    print("说明：真实模式(evaluate.py) 时，conf/risk 来自基座 logprob + 意图识别，")
    print('门控逻辑相同，仅信号来源换成真实模型。避免让护栏「每次都澄清」的抖动连续多数确认已内置。')
    return 0


def run_online(gate_cfg) -> int:
    from tfd.base.backends import build_backend
    backend = build_backend(load_config("base"))
    backend.load_model()
    print("已加载基座。在线双工+护栏闭环请在 AutoDL 上配合 50_record_demo.py 使用。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="用合成输入离线演示护栏逻辑")
    args = ap.parse_args()
    gate_cfg = load_config("gate")
    if args.offline:
        return run_offline(gate_cfg)
    return run_online(gate_cfg)


if __name__ == "__main__":
    raise SystemExit(main())