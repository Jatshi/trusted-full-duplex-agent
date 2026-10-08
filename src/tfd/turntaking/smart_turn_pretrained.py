"""Pinned, isolated adapter for Pipecat Smart Turn v3.2 CPU ONNX.

This is an upstream pretrained component, not a TFD-trained model. It only
estimates whether a *user* utterance is complete after VAD has found silence;
it does not classify backchannels, target speakers, or interruption intent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SAMPLE_RATE = 16_000
MAX_SAMPLES = 8 * SAMPLE_RATE


def prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Use the upstream last-eight-seconds, left-padding convention."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("expected nonempty, finite mono float audio at 16 kHz")
    if np.max(np.abs(samples)) > 1.01:
        raise ValueError("audio must be normalized to approximately [-1, 1]")
    if samples.size >= MAX_SAMPLES:
        return np.ascontiguousarray(samples[-MAX_SAMPLES:])
    return np.pad(samples, (MAX_SAMPLES - samples.size, 0)).astype(np.float32)


class SmartTurnV32:
    """Local CPU inference; intentionally not wired to the live demo by default."""

    def __init__(self, model_path: str | Path, *, verify_sha256: bool = True,
                 session=None, feature_extractor=None) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != MODEL_SHA256:
                raise ValueError(f"unexpected Smart Turn v3.2 model SHA256: {digest}")
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            # A sidecar must not monopolize CPUs needed by the live 9B service.
            options.intra_op_num_threads = 2
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if feature_extractor is None:
            from transformers import WhisperFeatureExtractor

            feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self.session = session
        self.feature_extractor = feature_extractor

    def probability_complete(self, audio: np.ndarray) -> float:
        samples = prepare_audio(audio)
        features = self.feature_extractor(
            samples, sampling_rate=SAMPLE_RATE, return_tensors="np",
            padding="max_length", max_length=MAX_SAMPLES,
            truncation=True, do_normalize=True,
        ).input_features.astype(np.float32)
        output = self.session.run(None, {"input_features": features})[0]
        probability = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Smart Turn output is not a finite sigmoid probability")
        return probability
