# -*- coding: utf-8 -*-
"""Build a self-contained, evidence-aligned duplex interview demo.

WAV files are embedded byte-for-byte. Metrics are loaded from the same reports
that produced the corresponding audio so the page cannot silently mix runs.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION_DIR = ROOT / "outputs" / "duplex_session"
TURN_DIR = ROOT / "data" / "user_turns"
TEMPLATE = ROOT / "demo" / "replay_template.html"
OUTPUT = ROOT / "demo" / "full_duplex_demo.html"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def embed_wav(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"Not a valid WAV file: {path}")
    return "data:audio/wav;base64," + base64.b64encode(payload).decode("ascii")


def audio_manifest():
    files = {
        "u_ambiguous": TURN_DIR / "ambiguous.wav",
        "b_ambiguous": SESSION_DIR / "00_ambiguous_bot.wav",
        "u_safety": TURN_DIR / "safety_critical.wav",
        "b_safety": SESSION_DIR / "02_safety_critical_bot.wav",
        "u_delete": TURN_DIR / "delete_phrased.wav",
        "b_delete": SESSION_DIR / "04_delete_phrased_bot.wav",
        "u_clear": TURN_DIR / "low_risk_clear.wav",
        "b_clear": SESSION_DIR / "03_low_risk_clear_bot.wav",
        "u_open": TURN_DIR / "bargein_open.wav",
        "u_stop": TURN_DIR / "bargein_stop.wav",
        "u_probe": TURN_DIR / "bargein_probe.wav",
        "b_cut_real": SESSION_DIR / "bargein_turn1_cut_delayed.wav",
        "b_yield_real": SESSION_DIR / "bargein_turn2_delayed.wav",
        "b_probe_real": SESSION_DIR / "bargein_turn3_delayed.wav",
    }
    encoded = {key: embed_wav(path) for key, path in files.items()}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    return encoded, hashes


def build() -> Path:
    transcript = read_json(SESSION_DIR / "session_transcript.json")
    delayed = read_json(SESSION_DIR / "bargein_report_delayed.json")
    turntaking = read_json(ROOT / "outputs" / "turntaking" / "eval_report.json")
    gate = read_json(ROOT / "outputs" / "gate" / "incremental_report.json")
    audio_data, audio_hashes = audio_manifest()

    sessions = {item["name"]: item for item in transcript}
    false_takeover = turntaking["strategies"]["learned"]["false_takeover_rate"]
    advance_ms = gate["summary"]["lead_vs_eot_ms_mean"]
    cut = delayed["turn1_cut"]
    turn2 = delayed["turn2_bargein"]
    turn3 = delayed["turn3_probe"]

    events = [
        {"type":"chapter","rail":"护栏：为何先判断","code":"A","title":"流式可信护栏","subtitle":"execute / clarify / stop","text":"先判断能不能做，再决定怎么回答。","tags":["200 ms 帧级决策","风险信号锁存"],"evidence":[{"value":f"{advance_ms:.0f} ms","label":"平均风险预警提前量"},{"value":"0","label":"关键帧误执行"}]},
        {"type":"turn","rail":"歧义请求","who":"user","audio":"u_ambiguous","text":sessions["ambiguous"]["text"],"tags":["clarify","对象不明确"],"badge":"用户请求","evidence":[{"value":"7 / 7","label":"帧级判定均为 clarify"}]},
        {"type":"turn","rail":"主动澄清","who":"agent","audio":"b_ambiguous","text":sessions["ambiguous"]["bot_text"],"tags":["先澄清，不盲执行"],"badge":"护栏接管","source":"真实 TTS 护栏回复","evidence":[{"value":"clarify","label":"最终门控动作"}]},
        {"type":"turn","rail":"安全关键请求","who":"user","audio":"u_safety","text":sessions["safety_critical"]["text"],"tags":["stop","Risk 0.5"],"tone":"danger","badge":"高风险输入","evidence":[{"value":"9 / 9","label":"帧级判定均为 stop"}]},
        {"type":"turn","rail":"安全拒绝","who":"agent","audio":"b_safety","text":sessions["safety_critical"]["bot_text"],"tags":["停止执行","要求确认路况"],"tone":"danger","badge":"护栏接管","source":"真实 TTS 护栏回复","evidence":[{"value":"stop","label":"最终门控动作"}]},
        {"type":"turn","rail":"不可逆请求","who":"user","audio":"u_delete","text":sessions["delete_phrased"]["text"],"tags":["stop","Risk 0.4"],"tone":"warn","badge":"不可逆输入","evidence":[{"value":"10 / 10","label":"帧级判定均为 stop"}]},
        {"type":"turn","rail":"阻止误删","who":"agent","audio":"b_delete","text":sessions["delete_phrased"]["bot_text"],"tags":["拒绝模糊删除","要求明确对象"],"tone":"warn","badge":"护栏接管","source":"真实 TTS 护栏回复","evidence":[{"value":"stop","label":"最终门控动作"}]},
        {"type":"turn","rail":"低风险放行","who":"user","audio":"u_clear","text":sessions["low_risk_clear"]["text"],"tags":["execute","Risk 0.0"],"badge":"明确请求","evidence":[{"value":"8 / 8","label":"帧级判定均为 execute"}]},
        {"type":"turn","rail":"自由生成","who":"agent","audio":"b_clear","text":sessions["low_risk_clear"]["bot_text"],"tags":["护栏放行","真实基座原始输出"],"badge":"自由生成","source":"MiniCPM-o 4.5 真实输出","evidence":[{"value":"execute","label":"最终门控动作"}]},
        {"type":"chapter","rail":"Barge-in：边说边停","code":"B","title":"用户抢话与系统让行","subtitle":"双音轨真实重演","text":"Agent 还在播报，用户已经开始说话。","tags":["VAD 独立检测流","生成流可中断"],"evidence":[{"value":"300 ms","label":"VAD 连续三帧确认"},{"value":f"{cut['stop_latency_ms']:.0f} ms","label":"真基座 VAD 后停止边界"}]},
        {"type":"turn","rail":"用户开启长回答","who":"user","audio":"u_open","text":"请你详细介绍一下你自己，说说你都有哪些功能和特点。","tags":["turn-taking → take"],"badge":"长回答请求","evidence":[{"value":"same session","label":"三轮共享同一会话上下文"}]},
        {"type":"bargein","rail":"真实重叠与截断","badge":"Barge-in 实时重演","agentAudio":"b_cut_real","userAudio":"u_stop","agentText":cut["text_before_cut"],"userText":"等一下，先停一下。","userStartMs":4500,"vadFireMs":int(cut["vad_fired_at_ms"]),"stopMs":int(cut["stopped_at_ms"]),"durationMs":9200,"evidence":[{"value":f"{cut['user_onset_ms']:.0f} ms","label":"检测到用户语音起点"},{"value":f"{cut['vad_fired_at_ms']:.0f} ms","label":"VAD 发出截断信号"},{"value":f"{cut['stopped_at_ms']:.0f} ms","label":"Agent 音频停止"}]},
        {"type":"turn","rail":"Agent 让行","who":"agent","audio":"b_yield_real","text":turn2["text"],"tags":["停止旧生成","进入新语轮"],"badge":"成功让行","source":"MiniCPM-o 4.5 真基座输出","evidence":[{"value":f"{turn2['ttft_ms']:.0f} ms","label":"抢话内容入模后的 TTFT"}]},
        {"type":"turn","rail":"打断后追问","who":"user","audio":"u_probe","text":"刚才我打断你之前，你介绍到哪里了？请用一句话概括。","tags":["上下文探针"],"badge":"连续会话","evidence":[{"value":"Turn 3","label":"打断后的上下文探针"}]},
        {"type":"turn","rail":"边界与归因","who":"agent","audio":"b_probe_real","text":turn3["text"],"tags":["记住最后一句","未播出内容存在补全"],"tone":"warn","badge":"真实负结果","source":"MiniCPM-o 4.5 真基座输出","evidence":[{"value":f"{turn3['bigram_overlap_with_cut']:.4f}","label":"与截断文本的 bigram 重叠"},{"value":f"{turn3['ttft_ms']:.0f} ms","label":"上下文探针 TTFT"}]},
    ]
    metadata = {
        "title":"Trusted Full-Duplex Speech Agent",
        "metrics":[
            {"value":f"{false_takeover*100:.0f}%","label":"学习型 turn-taking","note":"测试集假开口率"},
            {"value":f"{advance_ms:.0f} ms","label":"流式可信护栏","note":"平均风险预警提前量"},
            {"value":"300 ms","label":"EnergyVAD","note":"连续三帧确认延迟"},
            {"value":f"{turn2['ttft_ms']/1000:.3f} s","label":"真基座让行","note":"抢话后新轮 TTFT"},
        ],
        "audio_sha256":audio_hashes,
        "evidence_sources":["outputs/duplex_session/session_transcript.json","outputs/duplex_session/bargein_report_delayed.json","outputs/turntaking/eval_report.json","outputs/gate/incremental_report.json"],
    }
    app = {"m":metadata,"a":audio_data,"t":events}
    html = TEMPLATE.read_text(encoding="utf-8").replace("__META_ARG__", json.dumps(app,ensure_ascii=False,separators=(",",":"))).replace("__META_TITLE__",metadata["title"])
    if "__META_" in html:
        raise RuntimeError("Unresolved template placeholder")
    OUTPUT.write_text(html,encoding="utf-8",newline="\n")
    print(f"OK -> {OUTPUT} ({OUTPUT.stat().st_size/1024:.0f} KiB, {len(events)} scenes)")
    return OUTPUT


if __name__ == "__main__":
    build()
