#!/usr/bin/env python3
"""GPU check: capturing features must not change native duplex logits/cache length."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from bootstrap import ROOT  # noqa: F401 - installs src/ for direct script execution
from tfd.duplex_policy.feature_tap import StreamFeatureTap


def _load_extractor():
    path = Path(__file__).with_name("75_extract_minicpmo_duplex_features.py")
    spec = importlib.util.spec_from_file_location("tfd_minicpmo_extractor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(model, chunk: np.ndarray) -> dict:
    duplex = model.duplex
    prepare_kwargs = {
        "prefix_system_prompt": "<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
        "suffix_system_prompt": "<|audio_end|><|im_end|>",
    }

    def prefill(*, with_tap: bool):
        torch.manual_seed(173)
        torch.cuda.manual_seed_all(173)
        np.random.seed(173)
        model.reset_session(reset_token2wav_cache=True)
        if model.audio_past_key_values is not None:
            raise RuntimeError("parent audio KV cache did not reset")
        model.duplex_prepare(**prepare_kwargs)
        if with_tap:
            with StreamFeatureTap(duplex.decoder) as tap:
                tap.arm()
                result = model.duplex_prefill(audio_waveform=chunk)
                capture = tap.pop()
        else:
            result = model.duplex_prefill(audio_waveform=chunk)
            capture = None
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"duplex prefill failed: {result}")
        if duplex.pending_logits is None:
            raise RuntimeError("native duplex did not produce pending logits")
        return duplex.pending_logits.detach().float().cpu().clone(), duplex.decoder.get_cache_length(), capture

    plain_logits, plain_cache, _ = prefill(with_tap=False)
    tapped_logits, tapped_cache, capture = prefill(with_tap=True)
    plain_repeat, plain_repeat_cache, _ = prefill(with_tap=False)
    tapped_repeat, tapped_repeat_cache, repeated_capture = prefill(with_tap=True)
    difference = float((plain_logits - tapped_logits).abs().max())
    baseline_repeat = float((plain_logits - plain_repeat).abs().max())
    tap_repeat = float((tapped_logits - tapped_repeat).abs().max())
    acoustic_repeat = float((capture.acoustic - repeated_capture.acoustic).abs().max())
    semantic_repeat = float((capture.semantic - repeated_capture.semantic).abs().max())
    second_difference = float((plain_repeat - tapped_repeat).abs().max())
    passed = bool(
        baseline_repeat <= 1e-5 and tap_repeat <= 1e-5
        and difference <= 1e-5 and second_difference <= 1e-5
        and len({plain_cache, tapped_cache, plain_repeat_cache, tapped_repeat_cache}) == 1
    )
    return {
        "format": "tfd-star-feature-tap-invariance-v3",
        "max_abs_logit_difference": difference,
        "max_abs_logit_difference_repeat": second_difference,
        "plain_repeat_difference": baseline_repeat,
        "tap_repeat_difference": tap_repeat,
        "acoustic_capture_repeat_difference": acoustic_repeat,
        "semantic_capture_repeat_difference": semantic_repeat,
        "plain_cache_length": int(plain_cache),
        "tapped_cache_length": int(tapped_cache),
        "plain_repeat_cache_length": int(plain_repeat_cache),
        "tapped_repeat_cache_length": int(tapped_repeat_cache),
        "acoustic_capture_shape": list(capture.acoustic.shape),
        "semantic_capture_shape": list(capture.semantic.shape),
        "passed": passed,
        "scope_note": "One native chunk/one waveform; not an end-to-end long-session cache test.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pt-path")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native 9B invariance check")
    extractor = _load_extractor()
    model = extractor._load_model(args)
    duplex = model.duplex
    samples = int(duplex.FIRST_CHUNK_MS * duplex.SAMPLE_RATE / 1000)
    audio, _ = librosa.load(args.audio_path, sr=duplex.SAMPLE_RATE, mono=True)
    if not len(audio):
        raise ValueError("audio file is empty")
    chunk = np.pad(audio[:samples], (0, max(0, samples - len(audio)))).astype(np.float32)
    report = compare(model, chunk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
