#!/usr/bin/env python3
"""62 · barge-in 打断响应评测（本地机制部分，无需 GPU）。

用真实 bot 播报音频（outputs/duplex_session/*.wav）+ 合成用户插话，
测可中断播报机制的关键延迟：
  detection_latency  用户开口 -> VAD 判定（sustain 帧 + chunk 量化）
  stop_latency       VAD 判定 -> 播放停止（当前 chunk 播完即停）
  total_latency      用户开口 -> 播放停止

并给出 chunk_ms 粒度（100/200/500ms）的延迟-开销权衡：
chunk 越小停得越快，但每 chunk 一次模型/IO 调用的开销越高。

产出：outputs/turntaking/bargein_eval.json
用法：python scripts/62_bargein_eval.py
"""
from __future__ import annotations
import json
import zlib
from pathlib import Path
import numpy as np

from bootstrap import load_config, ROOT
from tfd.utils import audio as au
from tfd.turntaking.bargein import EnergyVAD, InterruptiblePlayback, make_mic_stream

SR_PLAY = 24000   # bot TTS 输出采样率（MiniCPM-o 官方固定 24k）


def chunkify(pcm: bytes, chunk_ms: int, sr: int):
    n = int(round(chunk_ms / 1000 * sr)) * 2      # pcm16 -> 2 bytes/sample
    return [pcm[i:i + n] for i in range(0, len(pcm), n)]


def synth_bot_tracks(n: int = 3, dur_s: float = 6.0, sr: int = SR_PLAY,
                     seed: int = 42) -> list[tuple[str, bytes]]:
    """无真实播报音频时的确定性降级源。

    机制层延迟只依赖播报轨的时长与 chunk 边界（VAD 走的是独立 mic 流，
    不看 bot 音频内容），故合成轨不影响检测/停止延迟测量的有效性；
    真实 TTS 包络只在能量起伏统计上略有差异，报告以 bot_source 区分。
    """
    rng = np.random.default_rng(seed)
    tracks = []
    for i in range(n):
        t = np.linspace(0, 1, int(sr * dur_s))
        env = np.clip(0.6 + 0.25 * np.sin(2 * np.pi * (0.7 + 0.3 * i) * t), 0.05, 1.0)
        audio = (rng.standard_normal(env.size) * env).astype(np.float32)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        tracks.append((f"synthbot_{i}", pcm))
    return tracks


def main():
    cfg = load_config("turntaking")["bargein"]
    print("=" * 72)
    print("62 · barge-in 打断响应评测（可中断播报机制）")
    print("=" * 72)

    wavs = sorted((ROOT / "outputs" / "duplex_session").glob("*_bot.wav"))
    if wavs:
        bot_source = "real_model"
        tracks = [(w.name, au.read_wav_pcm16(w, target_sr=SR_PLAY)) for w in wavs]
        print(f"bot 播报源: {len(tracks)} 条真实模型语音 (duplex_session/)")
    else:
        bot_source = "synthetic_fallback"
        tracks = synth_bot_tracks()
        print(f"bot 播报源: 无真机产物，降级为 {len(tracks)} 条确定性合成轨"
              "（机制延迟测量不受影响，来源已在报告标注）")

    results = []
    for chunk_ms in cfg["chunk_ms_list"]:
        for bargein_at in cfg["bargein_at_ms_list"]:
            latencies = []
            det_lats = []
            stop_lats = []
            n_interrupted = 0
            for name, pcm in tracks:
                dur_ms = len(pcm) / 2 / SR_PLAY * 1000
                if bargein_at + 1500 > dur_ms:
                    continue                       # 插话点须落在播报内
                vad = EnergyVAD(threshold_db=cfg["vad_threshold_db"],
                                frame_ms=100,
                                sustain_frames=cfg["vad_sustain_frames"])
                pb = InterruptiblePlayback(vad, chunk_ms=chunk_ms)
                mic = make_mic_stream(bot_len_ms=dur_ms, bargein_at_ms=bargein_at,
                                      bargein_speech_ms=1500, sr=16000,
                                      frame_ms=100,
                                      rng_seed=zlib.crc32(name.encode()) % 2**31)
                res = pb.play(iter(chunkify(pcm, chunk_ms, SR_PLAY)), mic)
                if res.interrupted:
                    n_interrupted += 1
                    latencies.append(res.stopped_at_ms - res.bargein_at_ms)
                    det_lats.append(res.detection_latency_ms)
                    stop_lats.append(res.stop_latency_ms)
            if not latencies:
                continue
            row = {
                "chunk_ms": chunk_ms,
                "bargein_at_ms": bargein_at,
                "n_trials": len(latencies),
                "n_interrupted": n_interrupted,
                "detection_latency_ms_mean": round(float(np.mean(det_lats)), 1),
                "stop_latency_ms_mean": round(float(np.mean(stop_lats)), 1),
                "stop_latency_ms_max": round(float(np.max(stop_lats)), 1),
                "total_latency_ms_mean": round(float(np.mean(latencies)), 1),
                "total_latency_ms_max": round(float(np.max(latencies)), 1),
            }
            results.append(row)
            print(f"  chunk={chunk_ms:4d}ms 插话@{bargein_at:4d}ms: "
                  f"打断 {n_interrupted}/{row['n_trials']}, "
                  f"检测 {row['detection_latency_ms_mean']}ms + 停止 "
                  f"{row['stop_latency_ms_mean']}ms = "
                  f"总延迟 mean={row['total_latency_ms_mean']}ms "
                  f"max={row['total_latency_ms_max']}ms")

    # 按 chunk 粒度汇总
    by_chunk = {}
    for r in results:
        by_chunk.setdefault(r["chunk_ms"], []).append(r["total_latency_ms_mean"])
    summary = {str(k): round(float(np.mean(v)), 1) for k, v in sorted(by_chunk.items())}
    print(f"\n各 chunk 粒度的平均打断响应延迟: {summary}")
    print("（chunk 越小停得越快，但每 chunk 一次调用开销越高——工程权衡点）")

    out = ROOT / "outputs" / "turntaking" / "bargein_eval.json"
    payload = {
        "vad": {"threshold_db": cfg["vad_threshold_db"],
                "sustain_frames": cfg["vad_sustain_frames"],
                "frame_ms": 100},
        "bot_source": bot_source,
        "detail": results,
        "latency_by_chunk_ms": summary,
        "note": "本地机制评测：bot 播报（真实模型语音；无真机产物时为确定性"
                "合成降级，机制延迟测量不受影响）+ 合成插话。真基座打断后"
                "的 KV 上下文保持见远端 63_duplex_bargein.py",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[ok] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
