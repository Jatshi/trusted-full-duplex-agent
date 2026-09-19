"""Short end-to-end probe for the audio full-duplex runtime.

The probe exercises the same gateway -> worker -> backend protocol as the
browser. It is intentionally bounded and prints per-input wall time so it can
be used for compile warm-up and deployment verification without a long run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "tests/cases/common/ref_audio"
INPUT_DIR = ROOT / "tests/cases/common/user_audio"
SAMPLE_RATE = 16_000


def _first_wav(directory: Path, preferred_prefix: str = "") -> Path:
    candidates = sorted(directory.glob("*.wav"))
    if preferred_prefix:
        preferred = [path for path in candidates if path.name.startswith(preferred_prefix)]
        if preferred:
            return preferred[0]
    if not candidates:
        raise FileNotFoundError(f"no WAV files found under {directory}")
    return candidates[0]


def _load_mono_float32(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def _b64(audio: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).decode()


async def _wait_for_session(ws, init_payload: dict, timeout: float) -> str:
    init_sent = False
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        event_type = message.get("type")
        if event_type in {"session.queue_done", "queue_done"} and not init_sent:
            await ws.send(json.dumps({"type": "session.init", "payload": init_payload}))
            init_sent = True
        elif event_type == "session.created":
            return str(message.get("session_id") or "")
        elif event_type in {"error", "session.closed"}:
            raise RuntimeError(f"session setup failed: {message}")


async def _init_direct_backend(ws, init_payload: dict, timeout: float) -> tuple[str, dict]:
    """Initialize the backend protocol without the queueing gateway."""

    await ws.send(
        json.dumps(
            {
                "type": "session.init",
                "payload": {"mode": "full_duplex", **init_payload},
            }
        )
    )
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        event_type = message.get("type")
        if event_type == "session.created":
            return str(message.get("session_id") or ""), dict(message.get("metrics") or {})
        if event_type in {"error", "session.closed"}:
            raise RuntimeError(f"session setup failed: {message}")


async def run(args: argparse.Namespace) -> None:
    reference = _load_mono_float32(args.reference)
    utterance = _load_mono_float32(args.input)
    one_second = SAMPLE_RATE
    padded = np.concatenate([utterance, np.zeros(args.silence_chunks * one_second, dtype=np.float32)])
    chunks = [padded[i : i + one_second] for i in range(0, len(padded), one_second)]
    chunks = chunks[: args.chunks]
    if chunks and len(chunks[-1]) < one_second:
        chunks[-1] = np.pad(chunks[-1], (0, one_second - len(chunks[-1])))

    init_payload = {
        "system_prompt": "你是 TFD-STAR，请用自然、简短的中文回答。",
        "voice": {
            "ref_audio_base64": _b64(reference),
            "tts_ref_audio_base64": _b64(reference),
        },
        "config": {"length_penalty": 1.05, "max_kv_tokens": 8192},
    }

    observations: list[dict] = []
    async with websockets.connect(
        args.gateway,
        max_size=128 * 1024 * 1024,
        ping_interval=None,
        close_timeout=5,
    ) as ws:
        if args.direct_backend:
            session_id, session_metrics = await _init_direct_backend(ws, init_payload, args.timeout)
        else:
            session_id = await _wait_for_session(ws, init_payload, args.timeout)
            session_metrics = {}
        print(f"session={session_id} chunks={len(chunks)}")
        if session_metrics:
            print("SESSION_METRICS " + json.dumps(session_metrics, ensure_ascii=False))

        for index, chunk in enumerate(chunks, start=1):
            started = time.perf_counter()
            await ws.send(
                json.dumps(
                    {
                        "type": "input.append",
                        "input": {"audio": _b64(chunk), "max_slice_nums": 1},
                    }
                )
            )
            terminal = None
            metrics: dict = {}
            text = ""
            reason = None
            while terminal is None:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
                event_type = message.get("type")
                if event_type == "response.output.delta":
                    kind = message.get("kind")
                    metrics.update(message.get("metrics") or {})
                    reason = message.get("reason") or reason
                    if kind == "text":
                        text += str(message.get("text") or "")
                    elif kind in {"listen", "audio"}:
                        terminal = kind
                elif event_type in {"response.listen", "duplex.output.listen"}:
                    terminal = "listen"
                elif event_type in {"error", "session.closed"}:
                    raise RuntimeError(f"runtime failed at chunk {index}: {message}")

            wall_ms = (time.perf_counter() - started) * 1000
            observation = {
                "chunk": index,
                "kind": terminal,
                "wall_ms": round(wall_ms, 1),
                "infer_ms": metrics.get("wall_clock_ms"),
                "reason": reason,
                "mlp_action": metrics.get("tfd_turn_action"),
                "mlp_allow_speak": metrics.get("tfd_turn_allow_speak"),
                "mlp_force_listen": metrics.get("tfd_turn_mlp_force_listen"),
                "mlp_probs": [
                    metrics.get("tfd_turn_prob_hold"),
                    metrics.get("tfd_turn_prob_take"),
                    metrics.get("tfd_turn_prob_backchannel"),
                ],
                "mlp_voiced_frames": metrics.get("tfd_turn_voiced_frames"),
                "text": text[-80:],
            }
            observations.append(observation)
            print(json.dumps(observation, ensure_ascii=False))

        if not args.direct_backend:
            await ws.send(json.dumps({"type": "session.close", "reason": "probe_done"}))
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
            except Exception:
                pass

    walls = [float(item["wall_ms"]) for item in observations]
    warm = walls[1:] if len(walls) > 1 else walls
    summary = {
        "session": session_id,
        "count": len(observations),
        "listen": sum(item["kind"] == "listen" for item in observations),
        "audio": sum(item["kind"] == "audio" for item in observations),
        "first_ms": walls[0] if walls else None,
        "warm_mean_ms": round(float(np.mean(warm)), 1) if warm else None,
        "warm_p90_ms": round(float(np.percentile(warm, 90)), 1) if warm else None,
        "warm_max_ms": round(float(np.max(warm)), 1) if warm else None,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="ws://127.0.0.1:8006/v1/realtime?mode=audio")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--chunks", type=int, default=15)
    parser.add_argument("--silence-chunks", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--direct-backend",
        action="store_true",
        help="connect directly to /backend instead of using the queueing gateway",
    )
    args = parser.parse_args()
    args.reference = args.reference or _first_wav(REF_DIR)
    args.input = args.input or _first_wav(INPUT_DIR, preferred_prefix="000_user_audio0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
