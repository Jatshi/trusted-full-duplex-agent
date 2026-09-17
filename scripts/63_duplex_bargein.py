#!/usr/bin/env python3
"""63 · 真基座 barge-in：VAD 触发截断 + 打断后上下文保持（AutoDL 上跑）。

62 已测机制层延迟（能量 VAD 检测 + 可中断播放，本地）；本脚本把同一套
检测接到真基座 MiniCPM-o 上，回答模型层三个问题：
  1) 半双工 streaming_generate 模式下，bot 播报中途被 VAD 截断（放弃
     streaming_generate 迭代器）后，能否立刻 prefill 新用户轮继续会话
     （不崩不挂）——这是半双工架构下"打断后继续"的可行性验证
  2) 打断轮的响应 TTFT：抢话音频收尾 -> bot 重新开口
  3) 打断后上下文保持：探针问题"刚才我打断你之前你介绍到哪里了"，
     答案与被打断文本做字符 bigram 重叠度比对（heuristic，附原文供人工复核）

流程（同一 session_id，KV 上下文连续）：
  Turn1  bargein_open.wav -> bot 长播报；播报音频按虚拟实时轴推进，
         同轴喂真实抢话语音帧(bargein_stop.wav)给 EnergyVAD，
         VAD 触发即在当前 chunk 截断（放弃剩余生成）
  Turn2  bargein_stop.wav -> 抢话内容正式入模（半双工：检测流与模型输入流
         分离，检测只用了能量，模型此时才"听到"完整抢话），测新轮 TTFT
  Turn3  bargein_probe.wav -> 上下文探针，全量收完

产出：outputs/duplex_session/bargein_report.json + bargein_turn*.wav
用法：
  python scripts/63_duplex_bargein.py --offline   # FakeModel 验证接线（本地）
  python scripts/63_duplex_bargein.py             # 真基座（AutoDL）
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bootstrap import load_config, ROOT
from tfd.utils import audio as au
from tfd.eval.metrics import bigram_overlap
from tfd.turntaking.bargein import EnergyVAD, frame_db

BOT_SR = 24000     # MiniCPM-o TTS 输出采样率
MIC_FRAME_MS = 100


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def mic_frames_from_wav(wav_path: Path, in_sr: int = 16000,
                         frame_ms: int = MIC_FRAME_MS,
                         pad_silence_s: float = 20.0) -> list[np.ndarray]:
    """把抢话 wav 展开成等长麦克风帧流，其后补静音帧（bot 还在播时保持供帧）。"""
    pcm = au.read_wav_pcm16(wav_path, target_sr=in_sr)
    n = int(frame_ms / 1000 * in_sr) * 2
    frames = [np.frombuffer(pcm[i:i + n], dtype="<i2").astype(np.float32) / 32768.0
              for i in range(0, len(pcm), n)]
    frames.extend([np.zeros(int(in_sr * frame_ms / 1000), dtype=np.float32)]
                  * int(pad_silence_s * 1000 / frame_ms))
    return frames


# ----------------------------------------------------------------------
# 会话流程
# ----------------------------------------------------------------------
def push_user_wav(backend, sid: str, wav_path: Path, chunk_s: float):
    """把一条用户 wav 按 chunk 推流给后端。"""
    from tfd.base.backends import StreamChunk
    pcm = au.read_wav_pcm16(wav_path, target_sr=backend.IN_SR)
    n = int(round(chunk_s * backend.IN_SR)) * 2
    for i, c in enumerate(au.chunks(pcm, n)):
        backend.push_user_audio(
            sid, StreamChunk(kind="user_audio", audio=c, ts=round(i * chunk_s, 3)))


def play_until_interrupt(backend, sid: str, mic_frames: list[np.ndarray],
                         vad: EnergyVAD) -> dict:
    """Turn1：消费 bot 流式输出，按虚拟实时轴同步喂 mic 帧给 VAD，触发即截断。

    返回 {interrupted, text_before_cut, audio_before_cut, vad_fired_at_ms,
          user_onset_ms, bot_audio_played_ms, stop_latency_ms}
    """
    text, audio = "", b""
    t_ms = 0.0            # 虚拟播放时间轴
    mic_i = 0
    for c in backend.iter_bot_output(sid):
        if c.kind == "bot_text" and c.text:
            text += c.text
        if c.kind == "bot_audio" and c.audio:
            audio += c.audio
            dur_ms = len(c.audio) / 2 / BOT_SR * 1000
            chunk_end_ms = t_ms + dur_ms    # 当前 chunk 在虚拟轴上的结束时刻
            n_f = max(int(round(dur_ms / MIC_FRAME_MS)), 1)
            for _ in range(n_f):
                if mic_i >= len(mic_frames):
                    break
                mic = mic_frames[mic_i]
                mic_i += 1
                t_ms += MIC_FRAME_MS
                if vad.feed_db(frame_db(mic)):
                    fired = t_ms
                    onset = max(fired - vad.sustain_frames * MIC_FRAME_MS, 0.0)
                    played = len(audio) / 2 / BOT_SR * 1000
                    # 与 62 的可中断播放语义一致：已进缓冲的 chunk 播完才静音
                    return {
                        "interrupted": True,
                        "text_before_cut": text,
                        "audio_before_cut": audio,
                        "vad_fired_at_ms": fired,
                        "user_onset_ms": onset,
                        "bot_audio_played_ms": round(played, 1),
                        "stopped_at_ms": round(chunk_end_ms, 1),
                        "stop_latency_ms": round(chunk_end_ms - fired, 1),
                    }
    # bot 播完 VAD 仍未触发（抢话落在播报之后）
    return {
        "interrupted": False,
        "text_before_cut": text,
        "audio_before_cut": audio,
        "bot_audio_played_ms": round(len(audio) / 2 / BOT_SR * 1000, 1),
        "stop_latency_ms": None,
    }


def run_turn(backend, sid: str, wav_path: Path, chunk_s: float) -> dict:
    """完整一轮：推用户音频 -> 全量收 bot 输出，测 TTFT（首 chunk 时间戳）。"""
    push_user_wav(backend, sid, wav_path, chunk_s)
    text, audio, ttft_ms = "", b"", None
    for c in backend.iter_bot_output(sid):
        if ttft_ms is None:
            ttft_ms = round(c.ts * 1000, 1)
        if c.kind == "bot_text" and c.text:
            text += c.text
        if c.kind == "bot_audio" and c.audio:
            audio += c.audio
    return {"text": text.strip(), "audio": audio, "ttft_ms": ttft_ms}


# ----------------------------------------------------------------------
# 离线 FakeModel（接线验证，无 GPU）
# ----------------------------------------------------------------------
class FakeModel:
    """turn1 长播报（多块流式输出），turn2 简短应答，turn3 复述被打断内容。"""

    def __init__(self):
        self.n_gen = 0
        self.prefills = []

    @staticmethod
    def _tone(seconds: float, f: float = 220.0):
        t = np.linspace(0, seconds, int(24000 * seconds), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * f * t)).astype("float32")

    def streaming_prefill(self, session_id, msgs, omni_mode, is_last_chunk):
        c = msgs[0]["content"][0]
        self.prefills.append((msgs[0]["role"],
                              None if isinstance(c, str) else len(c),
                              is_last_chunk))

    def streaming_generate(self, session_id, **kw):
        self.n_gen += 1
        if self.n_gen == 1:
            text = "我是一个语音助手，我可以回答问题、设置闹钟、查询天气，还能控制智能家居设备。"
            for i in range(0, len(text), 10):
                yield self._tone(0.6), text[i:i + 10]
        elif self.n_gen == 2:
            yield self._tone(0.5, 180), "好的，我先停一下，您请说。"
        else:
            yield self._tone(0.5, 160), "刚才我正在介绍我可以回答问题、设置闹钟、查询天气。"


def make_offline_backend(base_cfg):
    from tfd.base.backends import MiniCPMOBackend
    b = MiniCPMOBackend(base_cfg)
    b._model = FakeModel()
    return b


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="不拉基座，FakeModel 验证接线")
    ap.add_argument("--bargein-delay-ms", type=int, default=0,
                    help="抢话前补静音毫秒数，推迟打断点（A/B 归因：区分"
                         "「截断太早没内容可回忆」与「audio-token 回忆弱」）")
    args = ap.parse_args()

    base_cfg = load_config("base")
    bar_cfg = load_config("turntaking")["bargein"]
    chunk_s = base_cfg.get("duplex", {}).get("chunk_seconds", 0.5)
    turns_dir = ROOT / "data" / "user_turns"
    out_dir = ROOT / "outputs" / "duplex_session"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("63 · 真基座 barge-in：VAD 截断 + 打断后上下文保持"
          + ("（离线接线验证）" if args.offline else ""))
    print("=" * 72)

    if args.offline:
        backend = make_offline_backend(base_cfg)
    else:
        from tfd.base.backends import build_backend
        backend = build_backend(base_cfg)
        backend.load_model()
        print("基座已加载，开始 barge-in 会话")

    for name in ("bargein_open", "bargein_stop", "bargein_probe"):
        if not (turns_dir / f"{name}.wav").exists():
            print(f"[err] 缺用户语音 {turns_dir / name}.wav（本地先跑 gen_user_turns_tts.py）")
            return 1

    sid = "bargein"
    backend.register(sid)
    report = {"mode": "offline" if args.offline else "real",
              "backend": getattr(backend, "backend_name", "?"),
              "vad": {"threshold_db": bar_cfg["vad_threshold_db"],
                      "sustain_frames": bar_cfg["vad_sustain_frames"],
                      "frame_ms": MIC_FRAME_MS}}

    # ---- Turn1：长播报 + VAD 截断 ----
    t0 = time.time()
    push_user_wav(backend, sid, turns_dir / "bargein_open.wav", chunk_s)
    vad = EnergyVAD(threshold_db=bar_cfg["vad_threshold_db"],
                    frame_ms=MIC_FRAME_MS,
                    sustain_frames=bar_cfg["vad_sustain_frames"])
    mic_frames = mic_frames_from_wav(turns_dir / "bargein_stop.wav")
    if args.bargein_delay_ms > 0:
        n_pad = int(args.bargein_delay_ms / MIC_FRAME_MS)
        silence = np.zeros(int(16000 * MIC_FRAME_MS / 1000), dtype=np.float32)
        mic_frames = [silence] * n_pad + mic_frames
        print(f"[A/B] 抢话推迟 {args.bargein_delay_ms}ms（bot 已播出更多内容后再打断）")
    cut = play_until_interrupt(backend, sid, mic_frames, vad)
    suffix = "_delayed" if args.bargein_delay_ms > 0 else ""
    au.write_wav(out_dir / f"bargein_turn1_cut{suffix}.wav", cut["audio_before_cut"],
                 sample_rate=BOT_SR)
    print(f"\n[Turn1] bot 播报被截断: {cut['interrupted']}")
    if cut["interrupted"]:
        print(f"  抢话起点(虚拟轴) {cut['user_onset_ms']:.0f}ms | VAD 触发 "
              f"{cut['vad_fired_at_ms']:.0f}ms | 播报停止延迟 "
              f"{cut['stop_latency_ms']:.0f}ms")
        print(f"  已播出 {cut['bot_audio_played_ms']:.0f}ms / 截断前文本: "
              f"「{cut['text_before_cut'][:60]}」")
    else:
        print(f"  （bot 先说完，未发生打断；播报 {cut['bot_audio_played_ms']:.0f}ms）")

    # ---- Turn2：抢话入模，测新轮 TTFT ----
    try:
        t2 = run_turn(backend, sid, turns_dir / "bargein_stop.wav", chunk_s)
        t2_ok = True
    except Exception as e:                     # 截断后 prefill 崩溃本身就是结论
        t2, t2_ok = None, False
        print(f"\n[Turn2] [FAIL] 打断后新轮 prefill/generate 异常: {type(e).__name__}: {e}")
    if t2_ok:
        au.write_wav(out_dir / f"bargein_turn2{suffix}.wav", t2["audio"], sample_rate=BOT_SR)
        print(f"\n[Turn2] 打断后新轮: TTFT {t2['ttft_ms']}ms | bot: 「{t2['text'][:60]}」")

    # ---- Turn3：上下文探针 ----
    if t2_ok:
        try:
            t3 = run_turn(backend, sid, turns_dir / "bargein_probe.wav", chunk_s)
            au.write_wav(out_dir / f"bargein_turn3{suffix}.wav", t3["audio"], sample_rate=BOT_SR)
            ov = bigram_overlap(cut["text_before_cut"], t3["text"])
            print(f"\n[Turn3] 探针回答: 「{t3['text'][:80]}」")
            print(f"  与打断前文本 bigram 重叠度: {ov:.3f} "
                  f"(>=0.15 视为上下文保持，heuristic)")
        except Exception as e:
            t3, ov = None, None
            print(f"\n[Turn3] [FAIL] 探针轮异常: {type(e).__name__}: {e}")
    else:
        t3, ov = None, None

    backend.close(sid)

    report["turn1_cut"] = {k: v for k, v in cut.items()
                           if k not in ("audio_before_cut",)}
    report["turn1_cut"]["text_before_cut"] = cut["text_before_cut"]
    report["turn2_bargein"] = ({"ttft_ms": t2["ttft_ms"], "text": t2["text"]}
                               if t2_ok else {"failed": True})
    report["turn3_probe"] = ({"ttft_ms": t3["ttft_ms"], "text": t3["text"],
                              "bigram_overlap_with_cut": round(ov, 4)}
                             if t3 is not None else {"failed": True})
    report["context_preserved_heuristic"] = (
        bool(ov is not None and ov >= 0.15) if cut["interrupted"] else None)
    report["bargein_delay_ms"] = args.bargein_delay_ms
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["note"] = (
        "半双工架构：VAD 检测流与模型输入流分离（检测只用能量，抢话音频在 "
        "Turn2 才完整入模）。真·边听边说需 duplex_server 模式（set_break 协议）。"
        "上下文保持为 heuristic 判定，请结合 bargein_turn3.wav 人工复核。")
    out = out_dir / f"bargein_report{suffix}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[ok] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
