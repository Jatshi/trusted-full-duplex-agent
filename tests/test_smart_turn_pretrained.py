from pathlib import Path

import numpy as np
import pytest

from tfd.turntaking.smart_turn_pretrained import MAX_SAMPLES, SmartTurnV32, prepare_audio


def test_prepare_audio_suffix_and_left_padding():
    short = prepare_audio(np.ones(16000, dtype=np.float32))
    assert len(short) == MAX_SAMPLES
    assert np.all(short[:-16000] == 0)
    assert np.all(short[-16000:] == 1)
    long = prepare_audio(np.arange(MAX_SAMPLES + 10, dtype=np.float32) / (MAX_SAMPLES + 10))
    assert long[0] == pytest.approx(10 / (MAX_SAMPLES + 10))


@pytest.mark.parametrize("audio", [np.array([]), np.array([[0.1]]), np.array([np.nan]),
                                   np.array([2.0])])
def test_prepare_audio_rejects_invalid_input(audio):
    with pytest.raises(ValueError):
        prepare_audio(audio)


def test_adapter_runs_upstream_feature_contract_without_eager_dependencies(tmp_path: Path):
    class Features:
        input_features = np.ones((1, 80, 800), dtype=np.float32)

    class Extractor:
        def __call__(self, audio, **kwargs):
            assert len(audio) == MAX_SAMPLES
            assert kwargs["sampling_rate"] == 16000
            return Features()

    class Session:
        def run(self, names, feed):
            assert feed["input_features"].shape == (1, 80, 800)
            return [np.array([[0.7]], dtype=np.float32)]

    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"fixture")
    adapter = SmartTurnV32(fake, verify_sha256=False, session=Session(),
                           feature_extractor=Extractor())
    assert adapter.probability_complete(np.ones(16000, dtype=np.float32)) == pytest.approx(0.7)


def test_model_hash_mismatch_fails_closed(tmp_path: Path):
    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA256"):
        SmartTurnV32(fake)
