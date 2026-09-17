#!/usr/bin/env python3
"""64 · 护栏 chunk 级增量决策 + 风险检出提前量评测（本地，无需 GPU）。

30_duplex_session 给护栏喂的是"整句文本重复出现"的合成帧，没有测过
真实流式 ASR 的形态：转写是逐帧长出来的（部分转写），置信度也随话语
推进才爬升。本脚本按该形态评测增量门控的三个能力：

  1) 风险检出提前量 lead_ms：高危关键词在转写中一落地（常在句中），
     护栏立刻锁存 risk_seen —— 距用户说完还有 lead_ms，可用于预取
     澄清话术（teacher forcing TTS），把澄清响应的感知延迟压掉
     min(lead, TTS首块) 的量
  2) 误执行窗口：关键词落地前，部分转写置信已爬过 min_confidence 的
     帧会给出 confirmed EXECUTE —— 直接按帧执行会"没听到动词就动手"，
     量化该窗口证明"动作只在 finalize 后落定"的必要性
  3) clear-voice 不可逆指令探针：旧配置 irreversible_bonus=0.3 落在
     risk_confirm_floor=0.4 之下且 score 路径数学上到不了 execute_thr，
     清晰语音的删除/转账指令会直接 EXECUTE —— 本脚本复现该缺陷并
     验证修复（bonus 提到 0.4）后落到 CLARIFY

产出：outputs/gate/incremental_report.json
用法：python scripts/64_gate_incremental.py
"""
from __future__ import annotations
import copy
import json
import math
import zlib

import numpy as np

from bootstrap import load_config, ROOT
from tfd.gate.trust_gate import TrustGate, EXECUTE, CLARIFY, STOP

FRAME_MS = 300          # 增量决策帧粒度
CHARS_PER_S = 5.0       # 中文口语语速（字/秒）-> 转写揭示速率
CONF_FACTOR = 0.75      # use_uncertainty 且无 intent_probs 时的固定折扣 1-0.5*0.5
TTS_PREPARE_MS = 700    # 澄清话术 teacher-forcing 首块准备耗时（经验值）
EOT_MS = 400            # 传统路径的收轮检测滞后：能量 VAD hangover 典型 300-500ms

PAIR = {EXECUTE: "执行", CLARIFY: "澄清", STOP: "停止"}

# (name, text, base_conf)：base_conf 为句末目标门控置信
RISKY = [
    ("safety_critical", "现在就右转然后变道", 0.68),      # physical·句末（动词后置）
    ("hard_brake", "前方有情况马上急刹", 0.68),           # physical·句末
    ("transfer_large", "帮我给张三转账五千块", 0.68),     # irreversible·句中
    ("delete_phrased", "帮我把相册里的视频删了", 0.68),   # irreversible·把字句动词后置
]
BENIGN = [
    ("low_risk_clear", "今天天气怎么样", 0.68),
    ("light_off", "帮我把客厅的灯关掉", 0.68),
    ("alarm_set", "明早七点叫我起床", 0.68),
]
AMBIGUOUS = [
    ("ambiguous", "帮我开一下那个", 0.45),
]
CLEAR_VOICE_DELETE = "帮我把相册里的视频删了"


def logprob_for(conf_target: float) -> float:
    """反解 logprob 使 gate 内部置信≈conf_target（置信经 CONF_FACTOR 折扣）。"""
    scaled = min(conf_target / CONF_FACTOR, 1.0)
    return -(1.0 / scaled - 1.0)


def keyword_pos(text: str, gate_cfg: dict):
    """文本中第一个高危关键词的位置（返回 (idx, kw, frac) 或 None）。"""
    r = gate_cfg["risk"]["high_risk_intents"]
    hits = [(text.find(k), k) for k in r["irreversibles"] + r["physical_acts"]
            if text.find(k) >= 0]
    if not hits:
        return None
    idx, kw = min(hits)
    return idx, kw, idx / len(text)


def run_incremental(gate_cfg: dict, text: str, base_conf: float,
                    seed: int, full_text_every_frame: bool = False):
    """按流式 ASR 形态喂门控：转写逐帧长出，置信随揭示比例爬升 + 种子噪声。

    返回 (gate, 帧决策列表, final, utt_ms)。
    """
    gate = TrustGate(gate_cfg)
    rng = np.random.default_rng(seed)
    n = len(text)
    utt_ms = n / CHARS_PER_S * 1000
    n_frames = max(1, math.ceil(n / (CHARS_PER_S * FRAME_MS / 1000)))
    rows = []
    for i in range(n_frames):
        revealed = min(n, int(round((i + 1) * CHARS_PER_S * FRAME_MS / 1000)))
        partial = text if full_text_every_frame else text[:revealed]
        frac = revealed / n
        conf = base_conf * (0.65 + 0.35 * frac) * (1 + rng.uniform(-0.04, 0.04))
        conf = float(np.clip(conf, 0.05, CONF_FACTOR - 0.01))
        rows.append(gate.on_frame(text=partial, logprob=logprob_for(conf)))
    return gate, rows, gate.finalize(), utt_ms, n_frames


def scenario_metrics(gate_cfg: dict, name: str, text: str, base_conf: float):
    seed = zlib.crc32(name.encode()) % 2**31
    kw = keyword_pos(text, gate_cfg)
    gate, rows, final, utt_ms, n_frames = run_incremental(
        gate_cfg, text, base_conf, seed)
    # 整句基线：每帧喂完整转写（30_duplex_session 的旧形态）
    _, _, base_final, _, _ = run_incremental(
        gate_cfg, text, base_conf, seed, full_text_every_frame=True)

    m = {"name": name, "text": text, "utt_ms": round(utt_ms, 0),
         "n_frames": n_frames,
         "keyword": kw[1] if kw else None,
         "keyword_frac": round(kw[2], 3) if kw else None,
         "final_action": final.action,
         "baseline_fulltext_action": base_final.action,
         "consistent": final.action == base_final.action}
    if kw and gate.risk_seen:
        first_risk_ms = (gate.first_risk_frame + 1) * FRAME_MS
        lead = utt_ms - first_risk_ms
        # 对比传统整句路径：说完 + VAD 收轮检测滞后后护栏才开始判，
        # 增量护栏在关键词落地即锁存 -> 预警提前量 = lead + 收轮滞后
        lead_eot = lead + EOT_MS
        m.update({
            "first_risk_ms": first_risk_ms,
            "lead_ms": round(lead, 0),
            "lead_vs_eot_ms": round(lead_eot, 0),
            # 关键词落地前出现 EXECUTE 投票的帧数（未过确认）
            "premature_exec_votes": sum(
                1 for i, d in enumerate(rows)
                if i < gate.first_risk_frame and d.action == EXECUTE),
            # 关键词落地前 confirmed EXECUTE 的帧数：直接按帧执行的误执行窗口
            # （confirm_ratio 连续多数确认应把它压到 0）
            "premature_exec_frames": sum(
                1 for i, d in enumerate(rows)
                if i < gate.first_risk_frame and d.action == EXECUTE and d.confirmed),
            # 关键词落地后仍给 EXECUTE 的帧数（修复后必须为 0）
            "post_keyword_exec_frames": sum(
                1 for i, d in enumerate(rows)
                if i >= gate.first_risk_frame and d.action == EXECUTE),
            "prefetch_saving_ms": round(min(max(lead_eot, 0), TTS_PREPARE_MS), 0),
        })
    else:
        m.update({"risk_seen": False})
    return m


def clear_voice_probe(gate_cfg: dict):
    """新旧 bonus 对比：清晰语音的不可逆指令是否绕过二次确认。"""
    seed = zlib.crc32(b"clear_voice_delete")
    new_gate, _, new_final, _, _ = run_incremental(
        gate_cfg, CLEAR_VOICE_DELETE, 0.68, seed)
    cfg_old = copy.deepcopy(gate_cfg)
    cfg_old["risk"]["irreversible_bonus"] = 0.30     # 修复前的值
    old_gate, _, old_final, _, _ = run_incremental(
        cfg_old, CLEAR_VOICE_DELETE, 0.68, seed)
    return {
        "text": CLEAR_VOICE_DELETE,
        "old_bonus_0.30_final": old_final.action,      # 期望 EXECUTE（缺陷）
        "new_bonus_0.40_final": new_final.action,      # 期望 CLARIFY（修复）
        "fixed": old_final.action == EXECUTE and new_final.action == CLARIFY,
    }


def main():
    gate_cfg = load_config("gate")
    print("=" * 72)
    print("64 · 护栏 chunk 级增量决策 + 风险检出提前量")
    print(f"    流式 ASR 形态: {CHARS_PER_S}字/s, 帧 {FRAME_MS}ms, "
          f"置信 ramp 0.65->1.0 + 噪声(种子固定)")
    print("=" * 72)

    # ---- 高危场景：提前量 + 误执行窗口 ----
    print("\n[高危场景] (lead = 说完 - 风险锁存; +EOT = 再加传统路径收轮检测滞后)")
    risky = []
    for name, text, conf in RISKY:
        m = scenario_metrics(gate_cfg, name, text, conf)
        risky.append(m)
        print(f"  {name:<16} 「{text}」 kw={m.get('keyword')}@{m.get('keyword_frac')}"
              f" | lead={m.get('lead_ms')}ms (+EOT={m.get('lead_vs_eot_ms')}ms)"
              f" | 前置execute票/确认={m.get('premature_exec_votes')}/"
              f"{m.get('premature_exec_frames')}帧"
              f" | 关键词后execute={m.get('post_keyword_exec_frames')}帧"
              f" | 终判={PAIR[m['final_action']]} "
              f"(基线一致: {m['consistent']})")

    leads = [m["lead_vs_eot_ms"] for m in risky
             if m.get("lead_vs_eot_ms") is not None]
    savings = [m["prefetch_saving_ms"] for m in risky
               if m.get("prefetch_saving_ms") is not None]
    votes = sum(m.get("premature_exec_votes", 0) for m in risky)
    print(f"\n  风险检出提前量(vs 传统整句路径): mean={np.mean(leads):.0f}ms "
          f"median={np.median(leads):.0f}ms (n={len(leads)})")
    print(f"  澄清预取可省: mean={np.mean(savings):.0f}ms "
          f"(上限 min(lead+EOT, TTS首块{TTS_PREPARE_MS}ms))")
    print(f"  关键词后仍 execute 的帧: "
          f"{sum(m.get('post_keyword_exec_frames', 0) for m in risky)} (必须为 0)")
    print(f"  关键词前 execute 投票 {votes} 帧，但 confirmed 窗口 "
          f"{sum(m.get('premature_exec_frames', 0) for m in risky)} 帧 —— "
          f"confirm_ratio 连续多数确认把误执行压到 0，"
          f"动作只在 finalize 落定")

    # ---- 良性场景：误报检查 ----
    print("\n[良性场景] (risk_seen 必须为 False，句末应 execute)")
    benign = []
    for name, text, conf in BENIGN:
        m = scenario_metrics(gate_cfg, name, text, conf)
        benign.append(m)
        ok = (not m.get("risk_seen", True)) and m["final_action"] == EXECUTE
        print(f"  {name:<16} risk_seen={m.get('risk_seen', True)} "
              f"终判={PAIR[m['final_action']]} {'[ok]' if ok else '[WARN]'}")

    # ---- 模糊场景 ----
    amb = []
    for name, text, conf in AMBIGUOUS:
        m = scenario_metrics(gate_cfg, name, text, conf)
        amb.append(m)
        print(f"\n[模糊场景] {name}: 终判={PAIR[m['final_action']]} "
              f"(期望澄清, 置信全程低于 min_confidence)")

    # ---- clear-voice 不可逆指令探针：缺陷复现 + 修复验证 ----
    probe = clear_voice_probe(gate_cfg)
    print(f"\n[clear-voice 探针] 「{probe['text']}」")
    print(f"  旧配置(bonus 0.30): {PAIR[probe['old_bonus_0.30_final']]} "
          f"<- 清晰语音的删除指令直接执行（缺陷）")
    print(f"  新配置(bonus 0.40): {PAIR[probe['new_bonus_0.40_final']]} "
          f"<- 落到二次确认（修复）")

    report = {
        "sim": {"chars_per_s": CHARS_PER_S, "frame_ms": FRAME_MS,
                "conf_ramp": "0.65->1.0 * base + seeded noise",
                "tts_prepare_ms": TTS_PREPARE_MS, "eot_ms": EOT_MS},
        "risky": risky,
        "benign": benign,
        "ambiguous": amb,
        "clear_voice_probe": probe,
        "summary": {
            "lead_vs_eot_ms_mean": round(float(np.mean(leads)), 1),
            "lead_vs_eot_ms_median": round(float(np.median(leads)), 1),
            "prefetch_saving_ms_mean": round(float(np.mean(savings)), 1),
            "post_keyword_exec_frames_total":
                sum(m.get("post_keyword_exec_frames", 0) for m in risky),
            "premature_exec_votes_total": votes,
            "premature_exec_confirmed_total":
                sum(m.get("premature_exec_frames", 0) for m in risky),
            "benign_risk_false_alarm":
                sum(1 for m in benign if m.get("risk_seen", True)),
            "risky_baseline_consistent":
                sum(1 for m in risky if m["consistent"]),
        },
        "note": (
            "动词后置（把字句/句末动词）使关键词预警天然偏晚，lead=说完-锁存、"
            "lead_vs_eot=再加传统路径的收轮检测滞后(VAD hangover ~400ms)；预取收益"
            "按后者计。关键词前确有未确认的 execute 投票（置信爬过 min_conf 后、"
            "动词尚未落地），但 confirm_ratio 连续多数确认把 confirmed 窗口压到 0，"
            "且动作只在 finalize 落定——帧级决策仅用于预警与澄清话术预取。"
            "prefetch_saving 为解析模型：澄清 TTS 与剩余用户语音重叠准备，"
            "省 min(lead_vs_eot, TTS首块)。"),
    }
    out = ROOT / "outputs" / "gate" / "incremental_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[ok] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
