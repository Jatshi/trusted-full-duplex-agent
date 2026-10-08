from types import SimpleNamespace

import pytest

from py_backend.tfd_native_action import build_native_action_bridge_from_env


def test_disabled_does_not_touch_model(monkeypatch):
    monkeypatch.delenv("TFD_NATIVE_ACTION_HEAD", raising=False)
    assert build_native_action_bridge_from_env(object()) is None


def test_missing_project_fails_explicitly_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TFD_NATIVE_ACTION_HEAD", str(tmp_path / "head.pt"))
    monkeypatch.setenv("TFD_PROJECT_ROOT", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        build_native_action_bridge_from_env(object())


def test_backend_disabled_calls_original_prefill(monkeypatch):
    monkeypatch.delenv("TFD_NATIVE_ACTION_HEAD", raising=False)
    from core.processors.pytorch_backend import PyTorchBackend
    backend = PyTorchBackend("unused", 0)
    seen = []
    view = SimpleNamespace(prefill=lambda **kwargs: seen.append(kwargs) or {"ok": True})
    backend.processor = SimpleNamespace(set_duplex_mode=lambda: view)
    assert backend.duplex_prefill(audio_waveform="audio") == {"ok": True}
    assert seen[0]["audio_waveform"] == "audio"


def test_backend_enabled_routes_prefill_not_generate(monkeypatch):
    from core.processors.pytorch_backend import PyTorchBackend
    backend = PyTorchBackend("unused", 0)
    calls = []
    bridge = SimpleNamespace(prefill=lambda op: calls.append("bridge") or op())
    backend.tfd_native_action_bridge = bridge
    monkeypatch.setenv("TFD_NATIVE_ACTION_HEAD", "explicit.pt")
    view = SimpleNamespace(prefill=lambda **kwargs: calls.append("prefill"),
                           generate=lambda **kwargs: kwargs)
    backend.processor = SimpleNamespace(set_duplex_mode=lambda: view)
    backend.duplex_prefill()
    assert calls == ["bridge", "prefill"]
    assert backend.duplex_generate(force_listen=True) == {"force_listen": True}
