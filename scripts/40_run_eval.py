#!/usr/bin/env python3
"""40 · 全双工指标评测。产出可写进简历的数字。

离线模式用合成时钟测 metrics（验证管线通）；真实模式接入基座后，
对 configs/eval.yaml 的 scenarios 逐项收集并生成报告 + 图。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from bootstrap import load_config, ROOT
from tfd.eval.metrics import (
    first_token_latency, barge_in_latency, misuse_rate,
    duplex_simultaneity, content_bleu,
)


def run_offline(eval_cfg) -> dict:
    print("=" * 60)
    print("全双工指标评测 (离线合成时钟，验证管线)")
    print("=" * 60)
    r = {}
    # 模拟一次会话的事件时间线
    input_push_ts, first_token_ts = 1.0, 1.42
    barge_in_ts, bot_pause_ts = 4.0, 4.18
    overlap, total = 3.2, 9.0
    ref, hyp = "帮我把明早七点的闹钟设一下", "设置了明早七点的闹钟"

    r["first_token_latency_s"] = first_token_latency(input_push_ts, first_token_ts)
    r["barge_in_latency_s"] = barge_in_latency(barge_in_ts, bot_pause_ts)
    r["misuse_rate"] = misuse_rate(misclassified=1, total=8)
    r["duplex_simultaneity"] = duplex_simultaneity(overlap, total)
    r["content_bleu"] = content_bleu(ref, hyp)

    for k, v in r.items():
        print(f"  {k:<24} {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    out = ROOT / "outputs" / "eval_synthetic.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f">> 已存: {out}")
    return r


def run_online(eval_cfg) -> dict:
    print("真实评测：请在 AutoDL 上把基座流式输出接到各 metrics 收集点。")
    print("离线跑通管线后，对照 configs/eval.yaml 的 scenarios 填充 inputs 即可。")
    return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="offline", choices=["offline", "online"])
    args = ap.parse_args()
    cfg = load_config("eval")
    (ROOT / "outputs" / "eval_reports").mkdir(parents=True, exist_ok=True)
    if args.mode == "offline":
        run_offline(cfg)
    else:
        run_online(cfg)