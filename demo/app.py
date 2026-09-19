# -*- coding: utf-8 -*-
"""Gradio demo —— Trusted Full-Duplex Speech Agent
三个 Tab：
  1) 实证回放  —— 真实跑出的全双工案例（护栏拦截 ×5 + barge-in 三段回合 ×2）
  2) 实时决策  —— 现场麦克风 -> turn-taking / VAD 抢话 实时判定（CPU，无需 GPU）
  3) 真模型对话 —— 本地 MiniCPM-o 直连 (local_ssr) 或 远程 duplex_server (remote_ws)

运行（项目根目录）：
    python demo/app.py                  # 本机离线演示（Tab2/1 可用）
    python demo/app.py --share          # 生成公开链接（面试官浏览器直接开）
    python demo/app.py --backend local_ssr --config configs/base.yaml
    python demo/app.py --backend remote_ws --ws-url wss://xxx/ws/duplex/{session_id}
"""
from __future__ import annotations
import argparse
import base64
import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ))

import yaml  # noqa: E402
import gradio as gr  # noqa: E402

# 从 src/tfd 复用既有组件
from tfd.base.backends import (  # noqa: E402
    MiniCPMOBackend, MiniCPMODuplexServerBackend,
    StreamChunk, _b64_to_pcm16,
)
from tfd.gate.trust_gate import TrustGate  # noqa: E402
from tfd.turntaking.features import FrameFeaturizer, ENERGY_FLOOR_DB  # noqa: E402
from tfd.turntaking.predictor import (  # noqa: E402
    RulePauseStrategy, LearnedStrategy, BaseAggressiveStrategy,
    HOLD, TAKE, BACKCHANNEL,
)

CUR = Path(__file__).resolve().parent


# ============================================================================
# 配置加载
# ============================================================================
def load_cfg() -> dict:
    with open(CUR / "gradio_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["project_root"]).resolve()
    if not root.is_absolute():
        root = (CUR / root).resolve()
    cfg["_root"] = root
    # 证据优先：备份目录 > 项目 outputs
    ev = cfg.get("evidence", {})
    backup = (ev.get("backup_evidence_dir") or "").strip()
    if backup:
        _b = Path(backup)
        if _b.exists():
            ev["_evidence_dir"] = _b
    else:
        ev["_evidence_dir"] = root / "outputs" / "duplex_session"
    return cfg


# ============================================================================
# 通用小工具
# ============================================================================
def unpack_mic(audio):
    """gr.Audio(type='numpy') 输入是一个 (sr, data) 元组；这里统一解包。"""
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        return audio[0], audio[1]
    return None, audio


def on_live(cfg, audio):
    sr, data = unpack_mic(audio)
    return live_turntaking(cfg, sr, data)


def on_turn(args_cfg, cfg, audio):
    sr, data = unpack_mic(audio)
    return run_turn(args_cfg, cfg, sr, data)


def to_16k_mono(sr, data) -> np.ndarray:
    """把任意采样率/立体声的浏览器麦克风输入重采样到 16k 单声道。"""
    if data is None:
        return np.zeros(0, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.nan_to_num(data)
    if sr is None or int(sr) == 16000 or data.size == 0:
        return data
    n_out = int(round(data.size * 16000 / int(sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, data.size)
    x = np.linspace(0.0, 1.0, n_out)
    return np.interp(x, xp, data).astype(np.float32)


def float_to_pcm16(f16: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(f16), -1.0, 1.0) * 32767.0
    return arr.astype("<i2").tobytes()


def _audio_choice(wav_path: Path) -> Optional[str]:
    if wav_path and Path(wav_path).exists():
        return str(wav_path)
    return None


def _load_json(ev_dir: Path, name: str):
    p = ev_dir / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _reachable(path_str: str, root: Path, ev_dir: Path) -> Optional[str]:
    """录像 bot_wav 常为相对路径；找不到时尝试证据目录/项目根下解析。"""
    if not path_str:
        return None
    p = Path(path_str)
    cands = [p, ev_dir / p.name, root / p, root / p.name]
    for c in cands:
        if c.exists():
            return str(c)
    return None


# ============================================================================
# Tab1 · 实证回放
# ============================================================================
def _guardrail_cases(cfg) -> list[dict]:
    root = cfg["_root"]
    data = _load_json(root, cfg["evidence"]["transcript_json"])
    return data or []


def replay_choices(cfg) -> list[str]:
    cases = _guardrail_cases(cfg)
    return [f"{c.get('name')} · 护栏→{c.get('final')}" for c in cases]


def show_guardrail_case(cfg, label: str) -> tuple:
    cases = _guardrail_cases(cfg)
    ev_dir = cfg["evidence"]["_evidence_dir"]
    root = cfg["_root"]
    out = []
    match = [c for c in cases if label.startswith(c.get("name") + " ·")]
    c = match[0] if match else None
    if not c:
        return "", "", [], None
    frames = c.get("frames", [])
    action_seq = " ".join(f.get("action", "-") for f in frames)
    user = c.get("text", "")
    bot = c.get("bot_text", "")
    final = c.get("final")
    mode = c.get("bot_mode", "")
    risk = ""
    for f in frames:
        if f.get("risk"):
            risk += f" | chunk{f.get('chunk')}: risk={f.get('risk'):.2f}"
    header = (f"**用户指令：** {user}\n\n"
              f"**护栏最终决策：** `{final}`（{c.get('rationale','')}） · 播报方式 `{mode}`\n\n"
              f"**逐 chunk 决策：** `{action_seq}`{risk}\n\n"
              f"**机器人播报（护栏话术/回复）：** {bot}")
    audio = _audio_choice(_reachable(c.get("bot_wav"), root, ev_dir))
    return header, user, c, audio


def show_bargein_turn(cfg, frag: str, delayed: bool) -> tuple:
    """返回 (info_md, [a1, a2, a3])，三个音频槽固定。"""
    ev_dir = cfg["evidence"]["_evidence_dir"]
    suf = "_delayed" if delayed else ""
    rep = _load_json(ev_dir,
                     "bargein_report_delayed.json" if delayed else "bargein_report.json")
    if not rep:
        return "（barge-in 报告缺失）", [None, None, None]
    files = [f"bargein_turn1_cut{suf}.wav",
             f"bargein_turn2{suf}.wav", f"bargein_turn3{suf}.wav"]
    audio = [str(ev_dir / p) if (ev_dir / p).exists() else None for p in files]

    if frag == "回合语境":
        v = rep.get("vad", {})
        hdr = (
            f"**回放模式：** {rep.get('mode','')} / backend `{rep.get('backend','')}` · "
            f"VAD(threshold {v.get('threshold_db')}dB, sustain {v.get('sustain_frames')} 帧, frame {v.get('frame_ms')}ms)\n\n"
            f"**Turn1 截断：** `{rep['turn1_cut'].get('interrupted')}` · "
            f"截断前文本「{rep['turn1_cut'].get('text_before_cut','')}」· "
            f"停止延迟 **{rep['turn1_cut'].get('stop_latency_ms')}ms**\n\n"
            f"**Turn2 让行：** 「{rep['turn2_bargein'].get('text','')}」\n\n"
            f"**Turn3 探针（上下文保持）：** 「{rep['turn3_probe'].get('text','')}」 · "
            f"bigram 重叠 **{rep['turn3_probe'].get('bigram_overlap_with_cut','-')}**\n\n"
            f"`{rep.get('note','')}`"
        )
        return hdr, audio

    key = {"Turn1 截断": "turn1_cut", "Turn2 让行": "turn2_bargein",
           "Turn3 探针": "turn3_probe"}[frag]
    v = rep[key]
    if frag == "Turn1 截断":
        lines = (f"**是否被打断：** {v.get('interrupted')}\n\n"
                 f"**截断前文本：**「{v.get('text_before_cut','')}」\n\n"
                 f"**VAD 触发时刻：** {v.get('vad_fired_at_ms')}ms · "
                 f"用户说话起始 {v.get('user_onset_ms')}ms · "
                 f"已播 {v.get('bot_audio_played_ms')}ms\n\n"
                 f"**停止延迟：** {v.get('stop_latency_ms')}ms")
        keep = [audio[0], None, None]
    elif frag == "Turn2 让行":
        lines = (f"**新轮 TTFT：** {v.get('ttft_ms')}ms\n\n"
                 f"**让行话术：**「{v.get('text','')}」")
        keep = [None, audio[1], None]
    else:
        lines = (f"**TTFT：** {v.get('ttft_ms')}ms\n\n"
                 f"**探针回答：**「{v.get('text','')}」\n\n"
                 f"**与截断文本 bigram 重叠：** {v.get('bigram_overlap_with_cut','-')}")
        keep = [None, None, audio[2]]
    return lines, keep


# ============================================================================
# Tab2 · CPU 实时决策
# ============================================================================
def _build_strategy(cfg, name: str = None):
    name = name or cfg["tt"].get("strategy", "learned")
    w = cfg["tt"].get("learned_weights")
    if w:
        _w = cfg["_root"] / w
        w = str(_w) if _w.exists() else w
    if name == "learned":
        return LearnedStrategy(weights_path=w,
                               confirm_frames=cfg["tt"].get("confirm_frames", 4))
    if name == "rule_pause":
        return RulePauseStrategy(threshold_ms=500)
    return BaseAggressiveStrategy(threshold_ms=300)


def _turn_decisions(feats_seq: list[list[float]], strat):
    strat.reset()
    acts = []
    for f in feats_seq:
        acts.append(strat.feed(f))
    return acts


def live_turntaking(cfg, sr, data) -> str:
    audio = to_16k_mono(sr, data)
    if audio.size == 0:
        return "（没有收到音频）"
    frame_ms = cfg["tt"].get("frame_ms", 100)
    fn = int(round(frame_ms / 1000 * 16000))
    fz = FrameFeaturizer(frame_ms=frame_ms, vad_db=-38.0)
    frames = []
    for i in range(0, audio.size, fn):
        seg = audio[i:i + fn]
        if seg.size < fn:
            seg = np.pad(seg, (0, fn - seg.size))
        frames.append(seg)
    feats_seq = [fz.feed(fr) for fr in frames]

    learned = _build_strategy(cfg, "learned")
    rule = _build_strategy(cfg, "rule_pause")
    acts_l = _turn_decisions(feats_seq, learned)
    acts_r = _turn_decisions(feats_seq, rule)

    first_take_l = next((i for i, a in enumerate(acts_l) if a == TAKE), None)
    first_take_r = next((i for i, a in enumerate(acts_r) if a == TAKE), None)
    n_bc = acts_l.count(BACKCHANNEL)

    tl = f"第 {first_take_l} 帧 ({first_take_l * frame_ms}ms)" if first_take_l is not None else "未开口"
    trr = f"第 {first_take_r} 帧 ({first_take_r * frame_ms}ms)" if first_take_r is not None else "未开口"

    rows = []
    names = ["energy_db", "pause_len", "utt_speech_s", "pre_pause_speech_s", "pre_pause_drop"]
    n = len(acts_l)
    shown = min(n, 60)
    step = max(1, n // shown) if shown else 1
    for i in range(0, n, step):
        f = feats_seq[i]
        db = -ENERGY_FLOOR_DB * float(f[0]) + ENERGY_FLOOR_DB
        rows.append(
            f"| {i * frame_ms}ms | {db:.1f} | {f[1]*1.5:.2f}s | {f[5]*6.0:.2f}s | "
            f"`{acts_l[i]}` | `{acts_r[i]}` |"
        )
    table = "| 时刻 | 能量dB | 停顿 | 累计语音 | **学习型** | 规则停顿 |\n"
    table += "| --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows)

    return (
        f"**输入：** 采样率 {sr or '?'}Hz -> 重采样 16k，帧长 {frame_ms}ms，共 {n} 帧\n\n"
        f"**学习型决策器**（MLP, confirm={cfg['tt'].get('confirm_frames',4)}）：首次开口 **{tl}**；"
        f"backchannel `{n_bc}` 次\n\n"
        f"**规则停顿基线**：首次开口 **{trr}**\n\n"
        f"**逐帧判定：**\n{table}"
    )


# ============================================================================
# Tab3 · 真模型对话
# ============================================================================
_GB = {"backend": None, "mode": None, "lock": threading.Lock()}


def _build_backend(args_cfg, cfg):
    mode = args_cfg.get("backend", "auto")
    if mode in ("auto", "local_ssr"):
        try:
            base = yaml.safe_load(open(cfg["_root"] / args_cfg.get("config", "configs/base.yaml"), encoding="utf-8"))
            base["duplex"]["stream_style"] = "streaming_generate"
            return "local_ssr", MiniCPMOBackend(base)
        except Exception as e:  # noqa: BLE001
            if mode == "local_ssr":
                raise
    ws = args_cfg.get("ws_url") or cfg["backend"].get("ws_url")
    base2 = {"duplex": {"duplex_server": {"ws_url": ws,
                                          "session_prefix": cfg["backend"].get("session_prefix", "adx_"),
                                          "chunk_ms": cfg["backend"].get("chunk_ms", 1000),
                                          "system_prompt": cfg["backend"].get("system_prompt")}}}
    return "remote_ws", MiniCPMODuplexServerBackend(base2)


def _trust_gate(cfg):
    gp = cfg["_root"] / "configs/gate.yaml"
    gate = yaml.safe_load(open(gp, encoding="utf-8")) if gp.exists() else {}
    return TrustGate(gate)


def _init_backend(args_cfg, cfg, status: str) -> str:
    mode, backend = _build_backend(args_cfg, cfg)
    with _GB["lock"]:
        try:
            if _GB["backend"] is not None:
                for sid in list(_GB["backend"].sessions):
                    _GB["backend"].close(sid)
            backend.load_model()
            _GB["backend"], _GB["mode"] = backend, mode
        except Exception as e:  # noqa: BLE001
            return f"⚠️ 初始化失败：{e}"
    return (
        f"✅ 已连接：`{mode}`\n\n"
        f"{'本地直连 MiniCPM-o（需 GPU+权重）' if mode == 'local_ssr' else '远程 duplex_server（真·边听边说，可打断）'}。"
        f"\n现在可以：点 ⏺ 录音说一句话 -> 点 🗣 发送。"
    )


def _reset_backend(status: str) -> str:
    with _GB["lock"]:
        if _GB["backend"] is not None:
            try:
                for sid in list(_GB["backend"].sessions):
                    _GB["backend"].close(sid)
            except Exception:  # noqa: BLE001
                pass
        _GB["backend"], _GB["mode"] = None, None
    return "已断开连接。"


def _interrupt(status: str) -> str:
    with _GB["lock"]:
        b = _GB["backend"]
    if b is None:
        return "（尚未连接）"
    if not hasattr(b, "interrupt"):
        return "⚠️ 当前后端（local_ssr）不支持打断；请在远程 duplex_server 模式下测试打断。"
    try:
        sid = next(iter(b.sessions))
        b.interrupt(sid)
        return "⚠️ 已发送 set_break，模型已停下转向聆听。"
    except Exception as e:  # noqa: BLE001
        return f"打断失败：{e}"


def _drain_ws(backend, sid, timeout: float = 40.0) -> tuple:
    """远程 duplex_server：读结果，拆分 is_listen(用户 ASR) 与 bot 输出。"""
    bot_audio, bot_text, user_asr = b"", "", ""
    inbox = backend._inbox[sid]
    end = deadline = time.time() + timeout
    while time.time() < deadline and not end:
        try:
            msg = inbox.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg.get("type") != "result":
            continue
        if msg.get("is_listen"):
            user_asr += msg.get("text") or ""
        else:
            bot_text += msg.get("text") or ""
            a = msg.get("audio_data")
            if a:
                bot_audio += _b64_to_pcm16(a)
        if msg.get("end_of_turn"):
            end = True
    return bot_audio, bot_text, user_asr


def run_turn(args_cfg, cfg, sr, data) -> tuple:
    with _GB["lock"]:
        backend, mode = _GB["backend"], _GB["mode"]
    if backend is None:
        return ("（请先点「连接后端」）", None)
    audio = to_16k_mono(sr, data)
    if audio.size == 0:
        return ("（没有收到音频）", None)
    sid = uuid.uuid4().hex[:8]
    bot_audio, bot_text, user_asr, verdict = b"", "", "", ""
    try:
        backend.register(sid)
        if mode == "local_ssr":
            backend.push_user_audio(sid, StreamChunk(kind="user_audio",
                                                      audio=float_to_pcm16(audio)))
            for c in backend.iter_bot_output(sid):
                if c.kind == "bot_text":
                    bot_text += c.text
                elif c.kind == "bot_audio":
                    bot_audio += c.audio
        else:
            # 远程 ws：按 chunk_ms 切片发送，随后读结果
            chunk_n = int(16000 * cfg["backend"].get("chunk_ms", 1000) / 1000)
            pcm = float_to_pcm16(audio)
            for i in range(0, len(pcm), chunk_n * 2):
                backend.push_user_audio(sid, StreamChunk(kind="user_audio",
                                                         audio=pcm[i:i + chunk_n * 2]))
                time.sleep(0.03)
            bot_audio, bot_text, user_asr = _drain_ws(backend, sid)
            if user_asr.strip():
                gate = _trust_gate(cfg)
                d = gate.on_frame(text=user_asr.strip())
                verdict = (f"**护栏判定：** `{d.action}` · 风险 {d.risk:.2f}"
                           f" · 置信 {d.confidence:.2f}\n{d.rationale}")
        backend.close(sid)
    except Exception as e:  # noqa: BLE001
        return (f"⚠️ 回合失败：{e}", None)

    wav_path: Optional[str] = None
    if bot_audio:
        out = CUR / "tmp_mic" / f"turn_{sid}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        import wave
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(bot_audio)
        wav_path = str(out)

    transcript = f"**用户：**（现场录音{' · ' + user_asr if user_asr else ''}）\n\n"
    transcript += (f"**Bot：** {bot_text or '（无声频/文本）'}\n\n{verdict}"
                   if verdict else f"**Bot：** {bot_text or '（静音）'}")
    return transcript, wav_path


# ============================================================================
# Gradio 装配
# ============================================================================
def build_app(cfg, args_cfg):
    root = cfg["_root"]
    with gr.Blocks(title="Trusted Full-Duplex Speech Agent · Demo") as demo:

        gr.Markdown(
            f"# Trusted Full-Duplex Speech Agent · 交互 Demo\n"
            f"基座 **MiniCPM-o 4.5** · 系统层全双工 + 流式可信护栏 + turn-taking/barge-in + RL 对齐。"
            f"离线资源与实测证据见 [GitHub](https://github.com/Jatshi/trusted-full-duplex-agent)。"
        )

        with gr.Tabs():
            # ----------------- Tab3 真模型对话 -----------------
            with gr.Tab("🎙 真模型对话"):
                with gr.Row():
                    mode_md = gr.Markdown(f"**当前后端模式：** `{args_cfg.get('backend', 'auto')}`")
                with gr.Row():
                    btn_init = gr.Button("🔌 连接 / 初始化后端", variant="primary")
                    btn_break = gr.Button("⏹ 打断（remote_ws）")
                    btn_reset = gr.Button("断开")
                status = gr.Markdown("（未连接 —— 可先用 Tab1/2 的 CPU 演示）")
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy",
                               label="⏺ 录音（≤15秒）或上传音频文件 -> 点「发送这轮」",
                               streaming=False)
                btn_send = gr.Button("🗣 发送这轮", variant="primary")
                transcript = gr.Markdown("")
                bot_audio = gr.Audio(value=None, label="🔊 Bot 语音回复", type="filepath")
                with gr.Accordion("接线说明", open=False):
                    gr.Markdown(
                        "- **local_ssr**：本机 GPU + MiniCPM-o 权重，模型层半双工（流式生成，无法打断）。\n"
                        "- **remote_ws**：连 AutoDL 已启动的官方 `duplex_server`，真·边听边说，支持 `set_break` 打断。\n"
                        "- 当前为同步单轮交互（录音→回放），保证面试现场稳定；若要真·流式边听边说，请直接连 duplex_server 并在消费端持续推流。"
                    )
                btn_init.click(lambda: _init_backend(args_cfg, cfg, ""), outputs=status)
                btn_break.click(_interrupt, inputs=status, outputs=status)
                btn_reset.click(_reset_backend, inputs=status, outputs=status)
                btn_send.click(on_turn, inputs=[gr.State(args_cfg), gr.State(cfg), mic],
                               outputs=[transcript, bot_audio])

            # ----------------- Tab2 实时决策 -----------------
            with gr.Tab("⚡ 实时决策（现场麦克风）"):
                gr.Markdown(
                    "对着麦克风说一句带停顿的话（如「明天陪我…<停顿>…去爬山吧」），系统按 **200ms 帧**"
                    "实时给出 `hold / take / backchannel`（学习型 MLP）与规则停顿基线对比，并显示能量轨迹。"
                    "纯 CPU，无需 GPU。\n\n> 💡 公网录音若上传失败（无声/无反应），请直接把一段带停顿的音频"
                    "（≤15秒）拖进下方输入框的上传页再点「跑判定」，结论一致。"
                )
                mic2 = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                label="⏺ 录一段带停顿的话（≤15秒）或上传音频 -> 点「跑判定」",
                                streaming=False)
                btn_live = gr.Button("▶ 跑一遍实时判定", variant="primary")
                out_live = gr.Markdown("")
                btn_live.click(lambda a: on_live(cfg, a), inputs=mic2, outputs=out_live)

            # ----------------- Tab1 实证回放 -----------------
            with gr.Tab("📼 实证回放"):
                gr.Markdown(
                    "真实运行产出的案例音频（AutoDL·RTX 4080 SUPER 32GB / 9B MiniCPM-o）。"
                    "上半部为**护栏拦截**例，下半部为 **barge-in 三段回合**（原测试 vs 推迟打断 A/B 归因）。"
                )

                gr.Markdown("### A · 护栏拦截案例")
                guardrail_dd = gr.Dropdown(replay_choices(cfg),
                                           label="护栏案例（用户指令 → 护栏决策）", value=None)
                replay_info = gr.Markdown("")
                replay_audio = gr.Audio(value=None, label="🔊 机器人播报（该案例）",
                                        type="filepath")

                def on_guardrail(label):
                    h, _user, _c, a = show_guardrail_case(cfg, label)
                    return h, a
                guardrail_dd.change(on_guardrail, guardrail_dd,
                                    [replay_info, replay_audio])

                gr.Markdown("### B · barge-in 打断三段回合（A/B 归因）")
                with gr.Row():
                    b_group = gr.Dropdown(["原始测试（打断@500ms）",
                                           "延迟对照（打断@5000ms）"],
                                          label="A/B 组", value="原始测试（打断@500ms）")
                    b_frag = gr.Dropdown(["回合语境", "Turn1 截断", "Turn2 让行", "Turn3 探针"],
                                         label="回合片段", value="回合语境")
                b_info = gr.Markdown("")
                b_slots = [gr.Audio(value=None, label=l, type="filepath") for l in
                           ["Turn1 · 截断被掐", "Turn2 · 让行", "Turn3 · 探针"]]

                def on_bargein(group, frag):
                    delayed = "延迟" in group
                    h, aud = show_bargein_turn(cfg, frag, delayed)
                    return [h] + aud
                b_group.change(on_bargein, [b_group, b_frag], [b_info] + b_slots)
                b_frag.change(on_bargein, [b_group, b_frag], [b_info] + b_slots)

            # ----------------- SIDE 导航（Tab 外脚注） -----------------
        with gr.Row():
            gr.Markdown(
                "**数据流**：感知(帧提取+EnergyVAD) → 决策(turn-taking + 护栏) → "
                "执行(MiniCPM-o 流式) → 对齐(GRPO context_recall)。RNN 无，全为可复现实测。"
            )
        return demo


def main():
    ap = argparse.ArgumentParser(description="Trusted Full-Duplex Speech Agent demo")
    ap.add_argument("--backend", choices=["auto", "local_ssr", "remote_ws"], default="auto")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ws-url", default=None,
                    help="远程 duplex_server 地址，例如 wss://host:port/ws/duplex/{session_id}")
    ap.add_argument("--share", action="store_true", help="生成公开链接（面试用）")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    args_cfg = {"backend": args.backend, "config": args.config,
                "ws_url": args.ws_url or cfg["backend"].get("ws_url")}

    try:
        import gradio
        print(f"[demo] gradio={gradio.__version__}  backend={args.backend}")
    except Exception:  # noqa: BLE001
        pass
    demo = build_app(cfg, args_cfg)
    demo.launch(share=args.share, server_port=args.port,
                inbrowser=not args.no_browser,
                css="footer{display:none} .gradio-container{max-width:1100px!important}")


if __name__ == "__main__":
    main()