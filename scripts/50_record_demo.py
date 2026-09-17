#!/usr/bin/env python3
"""50 · 面试演示录制：跑一段对话，把输入/输出/护栏决策落成截图素材与 JSON。

目的：给你面试现场/录屏用的"输入显示 + 护栏各决策 + 最终决策"三栏素材。
若已跑过 30_duplex_session.py 且存在 outputs/duplex_session/session_transcript.json，
会优先并入真实基座的语音输出与护栏判定；否则退回合成演示。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from bootstrap import load_config, ROOT
from tfd.gate.trust_gate import TrustGate


DECISIONS = {"execute": "执行", "clarify": "澄清", "stop": "停止"}


DIALOG = [
    "帮我开一下那个",
    "今天天气怎么样",
    "现在就右转然后变道",
    "帮我把明早七点的闹钟设一下",
]


def _load_session():
    """若 30_duplex_session 已产出真实会话，返回其记录列表，否则 None。"""
    p = ROOT / "outputs" / "duplex_session" / "session_transcript.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def produce(gate_cfg) -> Path:
    gate = TrustGate(gate_cfg)
    real = _load_session()
    entries = []
    for turn, text in enumerate(DIALOG):
        # 若已匹配真实会话记录，直接采用其逐帧决策与最终判定（音画一致）；
        # 合成帧参数只是无真实数据时的兜底演示。
        hit = next((r for r in real if r.get("text") == text), None) if real else None
        if hit:
            frames = hit.get("frames", [])
            final_action, rationale = hit.get("final"), hit.get("rationale", "")
        else:
            low = any(k in text for k in ("那个", "右转", "变道"))
            frames = []
            for k in range(4):
                d = gate.on_frame(text=text, logprob=-0.6 if low else -0.08,
                                  asr_confidence=0.5 if low else 0.85)
                frames.append({"chunk": k, "action": d.action, "score": round(d.score, 3)})
            f = gate.finalize()
            final_action, rationale = f.action, f.rationale
        entry = {"turn": turn, "text": text, "frames": frames,
                 "final": final_action, "rationale": rationale}
        # 若存在真实基座会话，并入真实语音与护栏判定
        if hit:
            entry["real_bot_text"] = hit.get("bot_text", "")
            entry["real_bot_wav"] = hit.get("bot_wav", "")
            entry["real_final"] = hit.get("final")
            entry["bot_mode"] = hit.get("bot_mode", "")
            entry["source"] = "real_session"
        entries.append(entry)
        gate.reset()

    out = ROOT / "outputs" / "demo_transcript.json"
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    for e in entries:
        tag = f" | 真实音频: {e.get('real_bot_wav')}" if e.get("real_bot_wav") else ""
        mode = e.get("bot_mode")
        if mode == "guardrail_intercept":
            tag += " | 护栏拦截·teacher_forcing 播报"
        elif mode == "free_generation":
            tag += " | 放行·模型自由生成"
        print(f"turn{e['turn']}「{e['text']}」-> {e['final']} ({e['rationale']}){tag}")
    print(f"\n[ok] 演示素材已存: {out}")
    print("面试三栏可展示：用户输入 | 护栏逐帧决策 | 最终决策+理由"
          + ("（含真实基座语音）" if real else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-config", default=None)
    args = ap.parse_args()
    cfg = load_config("gate")
    if args.gate_config:
        import yaml
        from tfd.utils.config import deep_merge
        with open(args.gate_config, "r", encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, extra)
    produce(cfg)