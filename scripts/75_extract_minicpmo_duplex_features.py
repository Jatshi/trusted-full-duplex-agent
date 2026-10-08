#!/usr/bin/env python3
"""GPU extraction of native MiniCPM-o audio embeddings and causal hidden states.

This is the first script that intentionally requires AutoDL: it loads the 9B
backbone and replays annotated audio through the *native duplex* prefill path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from bootstrap import ROOT
from tfd.duplex_policy.feature_store import write_feature_record
from tfd.duplex_policy.feature_tap import (
    StreamFeatureTap,
    align_completed_chunks,
    causal_acoustic_frames,
    system_activity_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pt-path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/safeduplex_cn/features")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "data/safeduplex_cn/processed/features.jsonl")
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--allow-user-only-probe", action="store_true",
        help="Diagnostic only: zero-fill missing system track; force scientific_label=false",
    )
    return parser.parse_args()


def _load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.demo_root.resolve()))
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO

    model = MiniCPMO.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        _attn_implementation=args.attn_implementation,
    )
    config_path = Path(args.model_path) / "config.json"
    is_quantized = False
    if config_path.is_file():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        is_quantized = bool(config_payload.get("quantization_config"))
    if is_quantized:
        model.eval().cuda()
        print("[model] quantized checkpoint detected; preserving integer weights")
    else:
        model.bfloat16().eval().cuda()
    model.init_unified(
        pt_path=args.pt_path,
        preload_both_tts=False,
        duplex_config={"generate_audio": False, "ls_mode": "explicit"},
        device="cuda",
    )
    return model


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script on AutoDL")
    rows = _read_rows(args.manifest, args.limit)
    if not rows:
        raise ValueError("extraction manifest is empty")
    if args.output_manifest.exists():
        raise FileExistsError(
            f"refusing to append to existing manifest {args.output_manifest}; move or delete it explicitly"
        )
    for row in rows:
        if not row.get("system_audio_path") and not args.allow_user_only_probe:
            raise ValueError(
                f"{row['dialogue_id']}: separate system_audio_path required for full-duplex extraction"
            )
        if row.get("system_audio_path") and row.get("system_active_intervals") is None:
            raise ValueError(f"{row['dialogue_id']}: system_active_intervals must be reviewed")
    model = _load_model(args)
    torch.cuda.reset_peak_memory_stats()
    duplex = model.duplex
    extracted = []
    with StreamFeatureTap(duplex.decoder) as tap:
        for position, row in enumerate(rows, 1):
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = (args.manifest.parent / audio_path).resolve()
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            actual_duration_ms = int(round(len(waveform) / 16.0))
            if abs(actual_duration_ms - int(row["duration_ms"])) > 40:
                raise ValueError(f"{row['dialogue_id']}: annotation/audio duration mismatch")
            system_path_value = row.get("system_audio_path")
            if system_path_value:
                system_path = Path(system_path_value)
                if not system_path.is_absolute():
                    system_path = (args.manifest.parent / system_path).resolve()
                system_waveform, _ = librosa.load(system_path, sr=16000, mono=True)
                if abs(len(system_waveform) - len(waveform)) > 640:
                    raise ValueError(f"{row['dialogue_id']}: user/system track duration mismatch")
                # A <=40 ms recording offset is padding/trimming, not alignment
                # of independent microphones. True synchronization is a data requirement.
                system_waveform = np.pad(system_waveform[:len(waveform)],
                                         (0, max(0, len(waveform) - len(system_waveform))))
                activity = system_activity_frames(
                    row["system_active_intervals"], duration_ms=actual_duration_ms,
                    frame_ms=int(row["frame_ms"]),
                )
            else:
                system_waveform = np.zeros_like(waveform)
                activity = torch.zeros(
                    (int(np.ceil(actual_duration_ms / int(row["frame_ms"]))), 1),
                    dtype=torch.float32,
                )
            # duplex.prepare() resets its own decoder/processor but not the
            # parent MiniCPMO audio KV cache. Without this, dialogue N leaks
            # into N+1 even though the decoder cache length appears reset.
            model.reset_session(reset_token2wav_cache=True)
            if model.audio_past_key_values is not None:
                raise RuntimeError("parent audio KV cache did not reset")
            model.duplex_prepare(
                prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                suffix_system_prompt="<|audio_end|><|im_end|>",
            )
            captures, chunk_lengths_ms = [], []
            cursor = 0
            chunk_index = 0
            while cursor < len(waveform):
                chunk_ms = duplex.FIRST_CHUNK_MS if chunk_index == 0 else duplex.CHUNK_MS
                samples = int(chunk_ms * duplex.SAMPLE_RATE / 1000)
                chunk = waveform[cursor : cursor + samples]
                cursor += min(samples, len(waveform) - cursor)
                if len(chunk) < samples:
                    chunk = np.pad(chunk, (0, samples - len(chunk)))
                tap.arm()
                result = model.duplex_prefill(audio_waveform=chunk.astype(np.float32))
                if isinstance(result, dict) and result.get("success") is False:
                    raise RuntimeError(f"{row['dialogue_id']}: duplex prefill failed: {result}")
                capture = tap.pop()
                captures.append(capture)
                chunk_lengths_ms.append(int(chunk_ms))
                # Close each unit as LISTEN so the next prefix has a valid
                # native duplex KV-cache history. No TTS is generated.
                model.duplex_generate(force_listen_override=True)
                model.duplex_finalize()
                chunk_index += 1
            native_acoustic, semantic, available = align_completed_chunks(
                captures,
                chunk_ms=chunk_lengths_ms,
                duration_ms=actual_duration_ms,
                frame_ms=int(row["frame_ms"]),
            )
            raw_acoustic = causal_acoustic_frames(
                waveform, frame_ms=int(row["frame_ms"])
            )
            system_acoustic = causal_acoustic_frames(
                system_waveform, frame_ms=int(row["frame_ms"])
            )
            if not (len(raw_acoustic) == len(system_acoustic) == len(activity) == len(native_acoustic)):
                raise RuntimeError("user/system/playback/native frame clocks disagree")
            # User/system raw cues and playback state are causal at 160 ms.
            # Native embeddings and hidden states update only after full chunks.
            acoustic = torch.cat(
                [
                    raw_acoustic,
                    system_acoustic,
                    activity,
                    native_acoustic,
                    torch.tensor(available, dtype=torch.float32).unsqueeze(1),
                ],
                dim=1,
            )
            write_feature_record(
                feature_dir=args.output_dir,
                manifest_path=args.output_manifest,
                dialogue_id=row["dialogue_id"],
                acoustic=acoustic.numpy(),
                semantic=semantic.numpy(),
                actions=row["actions"],
                risk_labels=row["risk_labels"],
                frame_ms=int(row["frame_ms"]),
                take_event_index=row.get("take_event_index"),
                yield_event_index=row.get("yield_event_index"),
                source=row["source"],
                license_name=row["license"],
                condition={
                    **row.get("condition", {}),
                    "scientific_label": bool(row.get("condition", {}).get("scientific_label"))
                    and bool(system_path_value),
                    "user_only_probe": not bool(system_path_value),
                    "feature_alignment": "causal_dual_track_completed_chunk_hold_v2",
                },
                split_group=row.get("split_group", row["dialogue_id"]),
            )
            extracted.append(
                {
                    "dialogue_id": row["dialogue_id"],
                    "frames": int(len(acoustic)),
                    "acoustic_dim": int(acoustic.shape[1]),
                    "semantic_dim": int(semantic.shape[1]),
                    "duration_ms": actual_duration_ms,
                    "available_frames": int(sum(available)),
                    "unavailable_frames": int(len(available) - sum(available)),
                    "native_chunk_lengths_ms": chunk_lengths_ms,
                    "has_system_track": bool(system_path_value),
                }
            )
            print(f"[{position}/{len(rows)}] {row['dialogue_id']} -> {len(acoustic)} frames")
    report = {
        "format": "tfd-star-minicpmo-feature-extraction-v3",
        "model_path": str(Path(args.model_path).resolve()),
        "demo_root": str(args.demo_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "records": extracted,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "Feature extraction/probe metadata, not a model-quality result.",
        "alignment": "causal_dual_track_completed_chunk_hold_v2",
        "user_only_probe": any(not row.get("system_audio_path") for row in rows),
    }
    report_path = args.run_report or args.output_manifest.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] feature manifest -> {args.output_manifest}")
    print(
        f"[resources] peak_allocated={report['peak_allocated_gib']:.2f} GiB "
        f"peak_reserved={report['peak_reserved_gib']:.2f} GiB "
        f"elapsed={report['elapsed_seconds']:.1f}s"
    )
    print(f"[ok] extraction report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
