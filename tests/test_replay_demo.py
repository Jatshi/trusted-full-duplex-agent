import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def payload():
    html = (ROOT/"demo"/"full_duplex_demo.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="appdata" type="application/json">(.*?)</script>',html,re.S)
    assert match
    return html,json.loads(match.group(1))


def test_demo_is_self_contained():
    html,app = payload()
    assert "__META_" not in html
    assert "http://" not in html and "https://" not in html
    assert len(app["t"]) >= 12
    assert all(v.startswith("data:audio/wav;base64,") for v in app["a"].values())


def test_embedded_audio_is_byte_identical():
    _,app = payload()
    mapping={
        "u_stop":ROOT/"data"/"user_turns"/"bargein_stop.wav",
        "b_cut_real":ROOT/"outputs"/"duplex_session"/"bargein_turn1_cut_delayed.wav",
        "b_probe_real":ROOT/"outputs"/"duplex_session"/"bargein_turn3_delayed.wav",
    }
    for key,source in mapping.items():
        assert base64.b64decode(app["a"][key].split(",",1)[1]) == source.read_bytes()


def test_real_bargein_matches_report():
    _,app = payload()
    report=json.loads((ROOT/"outputs"/"duplex_session"/"bargein_report_delayed.json").read_text(encoding="utf-8"))
    scene=next(item for item in app["t"] if item["type"]=="bargein")
    cut=report["turn1_cut"]
    assert scene["vadFireMs"] == cut["vad_fired_at_ms"]
    assert scene["stopMs"] == cut["stopped_at_ms"]
    assert scene["agentText"] == cut["text_before_cut"]
    assert app["t"][-1]["text"] == report["turn3_probe"]["text"]
