import threading
import time
import numpy as np
import pytest
from tfd.duplex_policy.demo_observers import BoundedAudioObserver, parse_easy_turn, load_official_log_mel

def wait_until(check):
    end = time.monotonic()+2
    while not check():
        assert time.monotonic()<end
        time.sleep(.005)

def test_terminal_tag_only_and_invalid_not_coerced():
    assert parse_easy_turn(["你好<complete>"]) == "complete"
    assert parse_easy_turn(["嗯<backchannel>"]) == "backchannel"
    assert parse_easy_turn(["<wait><incomplete>"]) == "incomplete"
    assert parse_easy_turn(["<complete>但是"]) is None
    assert parse_easy_turn([]) is None

def test_actual_inference_and_audio_snapshot():
    seen=[]
    def infer(audio):
        seen.append(audio.copy())
        return {"p_complete":.7}
    obs=BoundedAudioObserver(infer, interval_samples=16000)
    obs.submit(np.ones(16000,dtype=np.float32)*.1, "a")
    wait_until(lambda: obs.metrics()["tfd_components_completed"] == 1)
    assert np.allclose(seen[0], .1)
    m=obs.metrics()
    assert m["tfd_components_result"]["p_complete"] == .7
    assert m["tfd_components_input_id"] == "a"
    assert m["tfd_components_controls_output"] is False
    obs.close()

def test_queue_bounded_latest_drop_and_buffer_cap():
    gate=threading.Event()
    entered=threading.Event()
    def infer(audio):
        entered.set(); gate.wait(2)
        return {"last":float(audio[-1])}
    obs=BoundedAudioObserver(infer, interval_samples=16000)
    obs.submit(np.zeros(16000,dtype=np.float32),"first")
    assert entered.wait(1)
    for i in range(12):
        obs.submit(np.ones(16000,dtype=np.float32)* (i/20),str(i))
    assert obs.pending.qsize() <= 1
    assert len(obs.audio) <= 8*16000
    assert obs.metrics()["tfd_components_dropped"] >= 10
    gate.set()
    wait_until(lambda: obs.metrics()["tfd_components_completed"] >= 2)
    assert obs.metrics()["tfd_components_input_id"] == "11"
    obs.close()

def test_close_discards_stale_result():
    gate=threading.Event(); entered=threading.Event()
    def infer(audio):
        entered.set(); gate.wait(2)
        return {"bad":"must_not_publish"}
    obs=BoundedAudioObserver(infer, interval_samples=16000)
    obs.submit(np.zeros(16000,dtype=np.float32),"old")
    assert entered.wait(1)
    obs.close(); gate.set()
    wait_until(lambda: not obs.thread.is_alive())
    assert obs.metrics()["tfd_components_completed"] == 0

@pytest.mark.parametrize("audio",[np.array([np.nan]),np.array([]),np.ones((2,2)),np.array([2.])])
def test_bad_audio_rejected_before_queue(audio):
    obs=BoundedAudioObserver(lambda audio: {},interval_samples=1)
    with pytest.raises(ValueError): obs.submit(audio,"bad")
    obs.close()

def test_failure_recorded_without_controlling_main():
    def fail(audio): raise RuntimeError("fixture failure")
    obs=BoundedAudioObserver(fail,interval_samples=1)
    obs.submit(np.zeros(1,dtype=np.float32),"one")
    wait_until(lambda: bool(obs.metrics()["tfd_components_error"]))
    assert obs.metrics()["tfd_components_controls_output"] is False
    assert obs.metrics()["tfd_components_completed"] == 0
    obs.close()

def test_exact_official_function_without_unrelated_import_side_effect(tmp_path):
    from types import SimpleNamespace
    source = tmp_path/"processor.py"
    source.write_text("raise RuntimeError('unrelated sox import')\ndef compute_log_mel_spectrogram(data):\n    yield torch.value + next(data)\n")
    func = load_official_log_mel(source, SimpleNamespace(value=4, nn=SimpleNamespace(functional=None)), None)
    assert next(func(iter([3]))) == 7
