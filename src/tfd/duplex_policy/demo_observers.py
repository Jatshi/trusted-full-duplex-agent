"""Bounded CPU observation of real microphone prefixes; never output control."""
import base64
import ast
import json
import queue
import re
import threading
import time
import urllib.request
import numpy as np

TAG = re.compile(r"<\s*(complete|incomplete|backchannel|wait)\s*>\s*(?:<\|endoftext\|>|<\|im_end\|>)?\s*$", re.I)

def load_official_log_mel(path, torch, librosa):
    # The official processor imports libsox solely for unrelated resampling.
    # Compile its exact log-mel function AST; do not replace it with fbank.
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "compute_log_mel_spectrogram"]
    if len(nodes) != 1:
        raise ValueError("Exactly one official log-mel function required")
    namespace = {"torch": torch, "librosa": librosa, "F": torch.nn.functional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["compute_log_mel_spectrogram"]

def parse_easy_turn(value):
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else ""
    match = TAG.search(value) if isinstance(value, str) else None
    return match.group(1).lower() if match else None

class BoundedAudioObserver:
    def __init__(self, infer, *, interval_samples=32000):
        self.infer = infer
        self.interval = interval_samples
        self.audio = np.empty(0, dtype=np.float32)
        self.samples_since = 0
        self.pending = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.closed = False
        self.completed = 0
        self.dropped = 0
        self.result = {}
        self.input_id = None
        self.error = None
        self.finished_at = None
        self.wall_ms = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, audio, input_id):
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.01:
            raise ValueError("Expected finite normalized mono 16k audio")
        if self.closed:
            return
        self.audio = np.concatenate((self.audio, wave))[-128000:]
        self.samples_since += len(wave)
        if self.samples_since < self.interval:
            return
        self.samples_since = 0
        job = (self.audio.copy(), input_id)
        try:
            self.pending.put_nowait(job)
        except queue.Full:
            try: self.pending.get_nowait()
            except queue.Empty: pass
            self.dropped += 1
            self.pending.put_nowait(job)

    def _work(self):
        while True:
            if self.closed:
                return
            try: wave, input_id = self.pending.get(timeout=.1)
            except queue.Empty: continue
            started = time.monotonic()
            try:
                result = self.infer(wave)
                if not isinstance(result, dict):
                    raise ValueError("Observer must return actual result dictionary")
                with self.lock:
                    if self.closed:
                        return
                    self.completed += 1
                    self.result = result
                    self.input_id = input_id
                    self.error = None
                    self.wall_ms = (time.monotonic()-started)*1000
                    self.finished_at = time.monotonic()
            except Exception as exc:
                with self.lock:
                    if not self.closed:
                        self.error = type(exc).__name__ + ": " + str(exc)[:180]

    def close(self):
        with self.lock:
            self.closed = True
        # Never block real-time main inference waiting on an observer.
        self.audio = np.empty(0, dtype=np.float32)

    def metrics(self):
        with self.lock:
            return {
                "tfd_components_enabled": True,
                "tfd_components_controls_output": False,
                "tfd_components_completed": self.completed,
                "tfd_components_dropped": self.dropped,
                "tfd_components_result": dict(self.result),
                "tfd_components_input_id": self.input_id,
                "tfd_components_result_age_ms": None if self.finished_at is None else (time.monotonic()-self.finished_at)*1000,
                "tfd_components_wall_ms": self.wall_ms,
                "tfd_components_error": self.error,
                "tfd_components_audio_scope": "rolling_8s_prefix_not_turn_segment_or_verified_near_end",
            }

def load_component_observer(url):
    # URL comes from server configuration, not user input; no external upload.
    if url != "http://127.0.0.1:22680":
        raise ValueError("Observer must use fixed loopback sidecar")
    with urllib.request.urlopen(url+"/health", timeout=5) as response:
        health = json.load(response)
    if health.get("status") != "ready" or health.get("device") != "cpu":
        raise RuntimeError("Actual component sidecar not ready")
    def infer(wave):
        payload = json.dumps({"audio":base64.b64encode(wave.tobytes()).decode(),"sample_rate":16000}).encode()
        req = urllib.request.Request(url+"/predict",data=payload,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=90) as response:
            result = json.load(response)
        probability = result.get("smart_turn_p_complete")
        if not isinstance(probability,(int,float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid Smart Turn prediction")
        if result.get("easy_turn_label") not in {None,"complete","incomplete","backchannel","wait"}:
            raise ValueError("Invalid EasyTurn tag")
        result["assets"] = health["assets"]
        return result
    return BoundedAudioObserver(infer)
