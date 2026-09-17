#!/usr/bin/env python3
"""30 · 端到端全双工会话（核心在线闭环，缺它项目就没"真跑"）。

把真实基座 + 流式可信护栏串成一条可跑的会话：
  用户音频(按 chunk 推流) -> backend.streaming_prefill
                              -> backend.streaming_generate (真实模型音频/文本)
                            每推一块同时喂 TrustGate 逐帧决策
                              -> 最终护栏判定 -> 保存真实语音 + 字幕 + 决策

用法：
  python scripts/30_duplex_session.py --offline        # 无 GPU 本地验证接线(合成)
  python scripts/30_duplex_session.py                  # AutoDL 上真跑(接基座)
  python scripts/30_duplex_session.py --audio-dir data/user_turns   # 用真实录音

说明：默认无录音时用测试信号占位，模型仍会真实生成；面试演示请把 16k 录音
放到 data/user_turns/<name>.wav（name 对 SCENARIOS）以获得真实人声输入。
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from bootstrap import load_config, ROOT
from tfd.gate.trust_gate import TrustGate
from tfd.gate.trust_gate import EXECUTE, CLARIFY, STOP
from tfd.utils import audio as au

# 与 20_run_gate_demo 保持一致的五类场景（text 视为用户句的 ASR 转写）
SCENARIOS = [
    ("ambiguous", "帮我开一下那个", "low"),
    ("high_risk_reversible", "帮我把明早七点的闹钟设一下", "low"),
    ("safety_critical", "现在就右转然后变道", "high"),
    ("low_risk_clear", "今天天气怎么样", "clear"),
    ("delete_phrased", "帮我把相册里的视频删了", "high"),
]
PAIR = {EXECUTE: "执行", CLARIFY: "澄清", STOP: "停止"}


@dataclass
class TurnRecord:
    name: str
    text: str
    frames: list = field(default_factory=list)
    final: str = CLARIFY
    rationale: str = ""
    bot_text: str = ""
    bot_wav: str = ""          # 相对 ROOT 的真实模型音频
    audio_source: str = ""
    bot_mode: str = ""         # free_generation | guardrail_intercept


# 护栏话术：clarify/stop 拦截后用 teacher_forcing 让模型逐字播报（非自由生成）
SPEECH_BY_ACTION = {
    CLARIFY: "不好意思，我没有完全听清您的意思。为了避免误操作，请您再具体说一次好吗？",
    STOP: "这个指令风险较高，我现在不能执行。请您先确认安全，再明确告诉我。",
}
SCENARIO_SPEECH = {
    "ambiguous": "不好意思，我没太听清。您是想让我打开哪个设备呢？",
    "high_risk_reversible": "好的，跟您确认一下：是要把闹钟设在明天早上七点整吗？",
    "safety_critical": "这个操作涉及行车安全，我现在不能执行。请先确认路况安全，再明确告诉我。",
    "delete_phrased": "删除操作不可恢复。请明确告诉我您要删除的具体内容，我再为您执行。",
}


def run_offline(gate_cfg) -> int:
    """无 GPU：用合成置信信号跑出同样的门控决策，验证会话脚本接线正确。"""
    print("=" * 72)
    print("30 · 端到端会话(离线合成，仅验证接线/决策，未拉基座)")
    print("=" * 72)
    gate = TrustGate(gate_cfg)
    for name, text, kind in SCENARIOS:
        low_conf = kind in ("low", "high")   # 模糊/高危 -> 置信低，触发澄清或停止
        for k in range(4):
            d = gate.on_frame(
                text=text,
                logprob=-0.6 if low_conf else -0.08,
                asr_confidence=0.5 if low_conf else 0.85,
            )
            print(f"  {name} chunk{k}: {PAIR[d.action]:<4} "
                  f"conf={d.confidence:.2f} risk={d.risk:.2f} score={d.score:.2f}")
        final = gate.finalize()
        print(f"  -> 最终: {PAIR[final.action]} ({final.rationale})")
        gate.reset()
    print("\n[ok] 接线就绪。上 AutoDL 去掉 --offline 即接真实基座。")
    return 0


def run_online(base_cfg, gate_cfg, audio_dir: str, out_dir: Path) -> int:
    from tfd.base.backends import build_backend

    backend = build_backend(base_cfg)
    backend.load_model()
    gate = TrustGate(gate_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(audio_dir) if audio_dir else (ROOT / "data" / "user_turns")
    records = []
    for idx, (name, text, kind) in enumerate(SCENARIOS):
        sid = f"turn_{idx}"
        backend.register(sid)
        # 1) 取用户音频：优先真实录音，否则测试信号占位
        wav_path = audio_dir / f"{name}.wav"
        if wav_path.exists():
            pcm = au.read_wav_pcm16(wav_path,
                                    target_sr=getattr(backend, "IN_SR", 16000))
            source = str(wav_path)
        else:
            pcm = au.make_test_signal(seconds=1.5,
                                      sample_rate=getattr(backend, "IN_SR", 16000))
            source = "synthetic-test-signal"
        # 2) 按 chunk 推流给后端 + 逐帧喂护栏
        chunk_s = base_cfg.get("duplex", {}).get("chunk_seconds", 0.5)
        in_sr = getattr(backend, "IN_SR", 16000)
        frames = []
        for i, c in enumerate(au.chunks(pcm, int(round(chunk_s * in_sr)) * 2)):
            backend.push_user_audio(sid, _mk_user_chunk(c, i, chunk_s))
            # 置信信号沿用 demo 的"模糊/高危 -> 低置信"（真实广告用 ASR logprob 替换）
            low_conf = kind in ("low", "high")
            d = gate.on_frame(
                text=text,
                logprob=-0.6 if low_conf else -0.08,
                asr_confidence=0.5 if low_conf else 0.85,
            )
            frames.append({"chunk": i, "action": d.action,
                           "score": round(d.score, 3),
                           "risk": round(d.risk, 3)})
        # 3) 护栏最终判定（生成前拦截点）：
        #    execute -> 放行，模型自由流式生成
        #    clarify/stop -> 拦截，teacher_forcing 把护栏话术逐字念出来，
        #                    话术作为真实 assistant 轮进入会话上下文
        final = gate.finalize()
        bot_text, bot_audio = "", b""
        if final.action == EXECUTE:
            bot_mode = "free_generation"
            for chunk in backend.iter_bot_output(sid):
                if chunk.kind == "bot_text" and chunk.text:
                    bot_text += chunk.text
                if chunk.kind == "bot_audio" and chunk.audio:
                    bot_audio += chunk.audio
        else:
            bot_mode = "guardrail_intercept"
            speech = SCENARIO_SPEECH.get(name, SPEECH_BY_ACTION[final.action])
            pv = backend.speak_preview(speech, sid=sid)
            bot_audio = pv.audio or b""
            bot_text = speech
        wav_rel = f"outputs/duplex_session/{idx:02d}_{name}_bot.wav"
        au.write_wav(ROOT / wav_rel, bot_audio,
                     sample_rate=getattr(backend, "OUT_SR", 24000))
        backend.close(sid)
        gate.reset()

        records.append(TurnRecord(
            name=name, text=text, frames=frames, final=final.action,
            rationale=final.rationale, bot_text=bot_text.strip(),
            bot_wav=wav_rel, audio_source=source, bot_mode=bot_mode,
        ))
        mode_tag = "护栏拦截·精确播报" if bot_mode == "guardrail_intercept" else "模型自由生成"
        print(f"\n[{idx}] 「{text}」 -> 护栏: {PAIR[final.action]} ({final.rationale})")
        print(f"    [{mode_tag}] bot: {bot_text[:80]!r}  wav: {wav_rel}  src: {source}")

    _dump(out_dir, records)
    print("\n[ok] 在线全双工会话完成。素材见 outputs/duplex_session/")
    return 0


def _dump(out_dir: Path, records: list[TurnRecord]):
    payload = [
        {"name": r.name, "text": r.text, "frames": r.frames, "final": r.final,
         "rationale": r.rationale, "bot_text": r.bot_text, "bot_wav": r.bot_wav,
         "audio_source": r.audio_source, "bot_mode": r.bot_mode}
        for r in records
    ]
    (out_dir / "session_transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mk_user_chunk(pcm: bytes, i: int, chunk_s: float):
    from tfd.base.backends import StreamChunk
    return StreamChunk(kind="user_audio", audio=pcm, ts=round(i * chunk_s, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="不拉基座，仅验证接线")
    ap.add_argument("--audio-dir", default=None, help="真实 16k 用户录音目录")
    args = ap.parse_args()
    base_cfg = load_config("base")
    gate_cfg = load_config("gate")
    out_dir = ROOT / "outputs" / "duplex_session"
    if args.offline:
        return run_offline(gate_cfg)
    return run_online(base_cfg, gate_cfg, args.audio_dir, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())