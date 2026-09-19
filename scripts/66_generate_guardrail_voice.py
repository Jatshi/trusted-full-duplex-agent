#!/usr/bin/env python3
"""Generate TrustGate responses with the same reference voice as the demo.

This is intentionally an offline deployment step.  Loading a second 9B model
next to the live duplex worker would waste VRAM, so operators stop the worker,
run this script once, and restart the service.  The resulting WAV files are
consumed by ``py_backend/server.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import ROOT, load_config
from tfd.base.backends import build_backend
from tfd.utils.audio import write_wav


RESPONSES = {
    "00_ambiguous_bot.wav": "为了避免误操作，我需要再确认一下。请把对象和具体操作说清楚。",
    "02_safety_critical_bot.wav": "这个指令可能带来安全风险，我现在不会执行。请先确认安全条件。",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "duplex_session",
    )
    args = parser.parse_args()

    ref_audio = args.ref_audio.expanduser().resolve()
    if not ref_audio.is_file():
        raise FileNotFoundError(f"reference audio not found: {ref_audio}")

    config = load_config("base")
    config["model"]["minicpm_o"]["ref_audio_path"] = str(ref_audio)
    backend = build_backend(config)
    backend.load_model()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in RESPONSES.items():
        chunk = backend.speak_preview(text)
        if not chunk.audio:
            raise RuntimeError(f"no audio generated for {filename}")
        output = args.output_dir / filename
        write_wav(output, chunk.audio, sample_rate=backend.OUT_SR)
        print(f"generated {output} with reference {ref_audio.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
