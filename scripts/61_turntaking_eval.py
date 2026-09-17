#!/usr/bin/env python3
"""61 · turn-taking 三策略事件级对比（held-out 测试集）。

对比口径（事件级，对齐 TurnGPT/VAP 的评估维度）：
  false_takeover_rate  抢话率：用户没说完就开口
  miss_rate            漏接率：说完 1s 内没接话
  take_latency_ms      接话延迟（中位数）
  backchannel_hit      犹豫停顿内发过反馈的比例 / 噪声反馈比例

策略：
  rule_pause        固定 500ms 停顿阈值（传统 VAD 方案）
  base_aggressive   模拟端到端基座内置策略（300ms+内容可答即开口，策略模拟非实测）
  learned           本项目训练的 MLP 决策器

产出：outputs/turntaking/eval_report.json
用法：python scripts/61_turntaking_eval.py  （需先跑 60 训练）
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from bootstrap import load_config, ROOT
from tfd.turntaking import (
    synth_dataset, frame_labels, featurize_utterance, evaluate_strategy,
    RulePauseStrategy, BaseAggressiveStrategy, LearnedStrategy, HOLD,
)


def run_suite(strategy, utts, cfg) -> dict:
    frame_ms = cfg["frame_ms"]
    events = []
    for u in utts:
        feats, _ = featurize_utterance(u, frame_ms=frame_ms)
        labels = frame_labels(u, frame_ms=frame_ms,
                              label_delay_s=cfg["label_delay_s"])
        if len(labels) < len(feats):
            labels = labels + [HOLD] * (len(feats) - len(labels))
        labels = labels[:len(feats)]
        ev = evaluate_strategy(
            strategy, feats, labels, frame_ms=frame_ms,
            miss_latency_ms=cfg["miss_latency_ms"],
            final_pause_start_ms=u.final_pause_start * 1000)
        events.append(ev)

    n = len(events)
    lat = [e.take_latency_ms for e in events
           if e.take_latency_ms is not None
           and e.take_latency_ms == e.take_latency_ms]   # 过滤 None/NaN
    n_filler = sum(1 for u in utts if u.filler_windows)
    return {
        "n": n,
        "false_takeover_rate": round(sum(e.false_takeover for e in events) / n, 4),
        "miss_rate": round(sum(e.miss for e in events) / n, 4),
        "take_latency_ms_median": float(np.median(lat)) if lat else None,
        "take_latency_ms_mean": round(float(np.mean(lat)), 1) if lat else None,
        "backchannel_hit": round(
            sum(e.backchannel_fired_in_filler for e in events) / max(n_filler, 1), 4),
        "backchannel_noise": round(
            sum(e.backchannel_fired_outside for e in events) / n, 4),
    }


def main():
    cfg = load_config("turntaking")
    d = cfg["data"]
    print("=" * 72)
    print("61 · turn-taking 三策略事件级对比（held-out test, "
          f"n={d['n_test']}, seed={d['seed']+2}）")
    print("=" * 72)
    ckpt = ROOT / "outputs" / "turntaking" / "learned_mlp.pt"
    if not ckpt.exists():
        print("[err] 先跑 60_turntaking_train.py 训练决策器")
        return 1

    test = synth_dataset(d["n_test"], seed=d["seed"] + 2)
    s = cfg["strategies"]
    frame_ms = cfg["frame_ms"]
    learned = LearnedStrategy(hidden=s["learned"]["hidden"], weights_path=str(ckpt))
    strategies = [
        ("rule_pause(500ms)", RulePauseStrategy(s["rule_pause"]["threshold_ms"])),
        ("base_aggressive(300ms)", BaseAggressiveStrategy(s["base_aggressive"]["threshold_ms"])),
        ("learned(MLP)", learned),
    ]

    # 特征缓存：confirm 扫描只影响决策规则，特征不变
    cache = []
    for u in test:
        feats, _ = featurize_utterance(u, frame_ms=frame_ms)
        labels = frame_labels(u, frame_ms=frame_ms,
                              label_delay_s=cfg["label_delay_s"])
        if len(labels) < len(feats):
            labels = labels + [HOLD] * (len(feats) - len(labels))
        labels = labels[:len(feats)]
        cache.append((feats, labels, u.final_pause_start * 1000))

    results = {}
    for name, strat in strategies:
        r = run_suite(strat, test, cfg)
        results[strat.name] = r
        print(f"\n[{name}]")
        print(f"  抢话率 false_takeover : {r['false_takeover_rate']*100:6.1f}%")
        print(f"  漏接率 miss            : {r['miss_rate']*100:6.1f}%")
        print(f"  接话延迟 median        : {r['take_latency_ms_median']:6.0f} ms")
        print(f"  backchannel 命中/噪声  : {r['backchannel_hit']*100:5.1f}% / "
              f"{r['backchannel_noise']*100:4.1f}%")

    # confirm_frames 超参扫描：抢话率-延迟 Pareto 前沿
    print("\nconfirm_frames 扫描（learned 策略的触发确认帧数）:")
    pareto = {}
    for cf in (1, 2, 3, 4, 5, 6):
        strat = LearnedStrategy(hidden=s["learned"]["hidden"],
                                 weights_path=str(ckpt), confirm_frames=cf)
        evs = [evaluate_strategy(strat, f, l, frame_ms=frame_ms,
                                 miss_latency_ms=cfg["miss_latency_ms"],
                                 final_pause_start_ms=fs)
               for f, l, fs in cache]
        n = len(evs)
        lat = [e.take_latency_ms for e in evs
               if e.take_latency_ms is not None
               and e.take_latency_ms == e.take_latency_ms]
        row = {
            "false_takeover_rate": round(sum(e.false_takeover for e in evs) / n, 4),
            "miss_rate": round(sum(e.miss for e in evs) / n, 4),
            "take_latency_ms_median": float(np.median(lat)) if lat else None,
        }
        pareto[cf] = row
        print(f"  confirm={cf}: 抢话 {row['false_takeover_rate']*100:5.1f}% | "
              f"漏接 {row['miss_rate']*100:4.1f}% | "
              f"延迟 {row['take_latency_ms_median']:5.0f}ms")

    out = ROOT / "outputs" / "turntaking" / "eval_report.json"
    payload = {
        "test_set": {"n_utterances": d["n_test"], "seed": d["seed"] + 2,
                     "note": "base_aggressive 为策略模拟（可复现），非基座实测"},
        "strategies": results,
        "confirm_frames_pareto": pareto,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[ok] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
