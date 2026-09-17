"""基座加载与流式 I/O 接口抽象。

设计：所有后端实现统一 DuplexBackend 接口，上层（护栏/RL/评测/demo）只依赖此接口，
不关心具体是 MiniCPM-o / VoiceChat / Moshi。默认推荐 MiniCPM-o 4.5。
"""
from __future__ import annotations
import abc
import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Iterator, Optional
import logging

logger = logging.getLogger("tfd.base")


@dataclass
class StreamChunk:
    """一个流式 chunk：可能是音频输入、音频输出、文本、或动作决定。"""
    kind: str            # "user_audio" | "bot_audio" | "user_text" | "bot_text" | "event"
    audio: Optional[bytes] = None   # pcm16/16k 或按具体后端约定
    text: Optional[str] = None
    ts: float = 0.0                 # 相对流开始的时间戳（秒）
    meta: dict = None


@dataclass
class DuplexSession:
    """一次"边听边说"会话的状态。"""
    sid: str
    overlapping_speaking: bool = False      # 是否正在并听并说
    user_has_floor: bool = True
    bot_has_floor: bool = False
    last_barge_in_ts: float = 0.0
    context: list = None

    def __post_init__(self):
        self.context = self.context or []


class DuplexBackend(abc.ABC):
    backend_name = "abstract"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.duplex_cfg = cfg.get("duplex", {})
        self.chunk_s = self.duplex_cfg.get("chunk_seconds", 0.5)
        self.sessions: dict[str, DuplexSession] = {}

    @abc.abstractmethod
    def load_model(self):
        """加载基座权重（在新机上第一次调用，之后走本地缓存）。"""

    @abc.abstractmethod
    def push_user_audio(self, sid: str, chunk: StreamChunk) -> None:
        """喂入用户音频 chunk。"""

    @abc.abstractmethod
    def iter_bot_output(self, sid: str) -> Iterator[StreamChunk]:
        """产出模型输出的流式 chunk（文本/音频/event）。"""

    @abc.abstractmethod
    def speak_preview(self, text: str, sid: Optional[str] = None) -> StreamChunk:
        """让 bot 把一条文本精确合成语音（护栏触发澄清/停止时的快路径）。

        传 sid 时在主会话上合成：护栏话术作为真实 assistant 轮进入对话上下文。
        """

    def register(self, sid: str) -> DuplexSession:
        s = DuplexSession(sid=sid)
        self.sessions[sid] = s
        return s

    def close(self, sid: str):
        self.sessions.pop(sid, None)


# ----------------------------------------------------------------------
# MiniCPM-o 4.5 实现（推荐）
# 官方流式范例（streaming_generate，半双工实时语音）：
#   输入：逐块  model.streaming_prefill(session_id, msgs=[{"role":"user","content":[audio16k]}],
#                                       omni_mode=False, is_last_chunk=...)
#   输出：iter_gen = model.streaming_generate(session_id, generate_audio, use_tts_template,
#                                             enable_thinking, do_sample, max_new_tokens,
#                                             length_penalty)
#         音频模式下 iter_gen 逐项产出 (wav_chunk_24k_tensor, text_chunk)
# 输入采样率固定 16000Hz，输出采样率固定 24000Hz。
# 真正的"听与说同时进行"走 duplex_server 官方 wss 服务（P2+），见下方实现。
# ----------------------------------------------------------------------
class MiniCPMOBackend(DuplexBackend):
    """MiniCPM-o 直接接线（stream_style='streaming_generate'，P0/P1）。"""
    backend_name = "minicpm_o"
    IN_SR = 16000     # 输入音频采样率（官方固定）
    OUT_SR = 24000    # 输出音频采样率（官方固定）

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._model = None
        model_cfg = cfg["model"]["minicpm_o"]
        self.repo = model_cfg["repo_or_path"]
        self.dtype = model_cfg.get("dtype", "bf16")
        self.load_from_local = model_cfg.get("load_from_local", False)
        self.weights_dir = cfg["paths"]["weights_dir"]
        self.device_map = model_cfg.get("device_map", "auto")
        self.attn_impl = model_cfg.get("attn_implementation", "sdpa")
        self.ref_audio_path = model_cfg.get("ref_audio_path")  # 可选：参考音频作声音克隆
        dup = cfg.get("duplex", {})
        self.stream_style = dup.get("stream_style", "streaming_generate")
        self.stream_cfg = dup.get("streaming_generate", {}) or {}
        self._session_prefix = self.stream_cfg.get("session_id_prefix", "tfd_")
        # 每个会话的运行时状态：待 prefill 的用户音频缓冲 + 生成流
        self._state: dict[str, dict] = {}

    # ---------------- 会话生命周期 ----------------
    def register(self, sid: str) -> DuplexSession:
        s = super().register(sid)
        self._state[sid] = {
            "pending": [],   # 已缓存但尚未 prefill 的 float32/16k 数组
            "samples": 0,    # 已累积采样点数
            "gen": None,     # 进行中的 streaming_generate 迭代器
            "had_input": False,  # 本轮是否推过用户音频
            "system_done": False,
        }
        return s

    def close(self, sid: str):
        self._state.pop(sid, None)
        super().close(sid)

    # ---------------- 模型加载 ----------------
    def load_model(self):
        if self._model is not None:
            return self._model
        import os
        import torch
        from transformers import AutoModel
        path = self.repo
        if self.load_from_local:
            cache = os.path.join(self.weights_dir, "minicpm-o-4_5")
            path = cache if os.path.isdir(cache) else self.repo
        torch_dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float16
        logger.info(f"loading MiniCPM-o from: {path} ({self.dtype})")
        self._model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            attn_implementation=self.attn_impl,   # "sdpa" 或 "flash_attention_2"
            torch_dtype=torch_dtype,
            init_vision=True,
            init_audio=True,
            init_tts=True,
            device_map=self.device_map,
        )
        self._model.eval()
        self._model.init_tts()
        # token2wav 音色缓存：官方裸调 streaming_generate 不会自动初始化（只有
        # as_duplex().prepare() 会做），未初始化时首次出音频即 NoneType 崩溃。
        # 未配置参考音频则用模型自带 assets/system_ref_audio.wav（默认音色）。
        ref_path = self.ref_audio_path
        if not (ref_path and os.path.exists(ref_path)):
            ref_path = os.path.join(path, "assets", "system_ref_audio.wav")
        if ref_path and os.path.exists(ref_path):
            import librosa
            ref, _ = librosa.load(ref_path, sr=self.IN_SR, mono=True)
            self._model.reset_session(reset_token2wav_cache=True)
            self._model.init_token2wav_cache(prompt_speech_16k=ref)
            logger.info(f"token2wav cache initialized from: {ref_path}")
        else:
            logger.warning("no reference audio for token2wav cache; TTS audio may fail")
        return self._model

    # ---------------- PCM 编解码 ----------------
    @staticmethod
    def _pcm16_to_float(pcm: bytes):
        import numpy as np
        arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        return arr / 32768.0

    @staticmethod
    def _float_to_pcm16(f16):
        import numpy as np
        if hasattr(f16, "detach"):          # torch.Tensor -> numpy
            f16 = f16.detach().cpu().numpy()
        arr = np.asarray(f16).reshape(-1)
        arr = np.clip(arr, -1.0, 1.0) * 32767.0
        return arr.astype("<i2").tobytes()

    def _sid_token(self, sid: str) -> str:
        return f"{self._session_prefix}{sid}"

    # ---------------- 用户输入：累积 + 增量 prefill ----------------
    def _ensure_system_prefill(self, sid: str):
        """官方流式范例要求：用户音频前先 prefill 系统轮（每会话一次）。"""
        st = self._state[sid]
        if st.get("system_done"):
            return
        sys_prompt = self.stream_cfg.get(
            "system_prompt",
            "你是一个乐于助人的语音助手，请用简洁自然的中文口语回答用户。",
        )
        self._model.streaming_prefill(
            session_id=self._sid_token(sid),
            msgs=[{"role": "system", "content": [sys_prompt]}],
            omni_mode=False,
            is_last_chunk=True,
        )
        st["system_done"] = True

    def push_user_audio(self, sid: str, chunk: StreamChunk) -> None:
        """把用户音频 chunk（pcm16/16k bytes）缓存到会话。

        为降低首 token 延迟，凡是够一个 1s prefill 块的音频会立即喂给模型；
        不足 1s 的残块保留，留待 iter_bot_output 末尾补零成 1s 再标记轮次结束。
        """
        if self._model is None:
            self.load_model()
        st = self._state.get(sid)
        if st is None:
            raise KeyError(f"unknown duplex session: {sid}")
        if not chunk.audio:
            return
        self._ensure_system_prefill(sid)
        st["had_input"] = True
        st["pending"].append(self._pcm16_to_float(chunk.audio))
        st["samples"] += int(st["pending"][-1].size)
        self._prefill_full_blocks(sid, mark_last=False)

    def _prefill_full_blocks(self, sid: str, mark_last: bool):
        """把缓冲中满 1s 的整块通过 streaming_prefill 喂给模型。"""
        import numpy as np
        st = self._state[sid]
        if not st["pending"]:
            return
        pending = np.concatenate(st["pending"], axis=0).astype(np.float32)
        step = self.IN_SR
        n_full, rem = divmod(pending.size, step)
        if n_full == 0:
            return
        head = pending[: n_full * step]
        tail = pending[n_full * step:] if rem else None
        for i in range(n_full):
            seg = head[i * step : (i + 1) * step]
            is_last = mark_last and (i == n_full - 1)
            self._model.streaming_prefill(
                session_id=self._sid_token(sid),
                msgs=[{"role": "user", "content": [seg]}],
                omni_mode=False,
                is_last_chunk=is_last,
            )
        st["pending"] = [tail] if tail is not None else []
        st["samples"] = int(tail.size) if tail is not None else 0

    # ---------------- 模型输出：流式生成 ----------------
    def iter_bot_output(self, sid: str) -> Iterator[StreamChunk]:
        """产出 bot 回复的流式 chunk（文本 + 音频）。

        每轮：先把残块补零到 1s 并标记 is_last_chunk=True 结束用户轮次，
        再调用一次 streaming_generate，逐 chunk yield bot_text / bot_audio。
        """
        if self._model is None:
            self.load_model()
        st = self._state[sid]
        t0 = time.time()
        # 1) 收尾用户轮次：把不足 1s 的残块补零到 1s 并标记最后一块，触发模型响应
        self._close_user_turn(sid)
        # 2) 流式生成并产出。官方两种迭代格式：
        #    generate_audio=True  -> (wav_chunk, text_chunk)
        #    generate_audio=False -> (text_chunk, is_finished)
        gen_audio = self.stream_cfg.get("generate_audio", True)
        try:
            stream = self._streaming_generate(sid)
            if gen_audio:
                for wav_chunk, text_chunk in stream:
                    ts = round(time.time() - t0, 3)
                    if text_chunk:
                        yield StreamChunk(kind="bot_text", text=text_chunk, ts=ts)
                    if wav_chunk is not None:
                        yield StreamChunk(
                            kind="bot_audio",
                            audio=self._float_to_pcm16(wav_chunk),
                            ts=ts,
                        )
            else:
                for text_chunk, _fin in stream:
                    if text_chunk:
                        yield StreamChunk(
                            kind="bot_text", text=text_chunk,
                            ts=round(time.time() - t0, 3))
        finally:
            st["pending"] = []
            st["samples"] = 0
            st["had_input"] = False

    def _close_user_turn(self, sid: str):
        """把残余块补零到 1s 作为最后一块 prefill，is_last_chunk=True。

        整秒音频（无残块）也要发一块补零块收尾，否则用户轮永不结束、
        模型不知道该开始说话。完全没输入时不发（避免对纯静音应答）。
        """
        import numpy as np
        st = self._state[sid]
        tail = np.concatenate(st["pending"], axis=0).astype(np.float32) if st["pending"] else None
        st["pending"] = []
        st["samples"] = 0
        if not st.get("had_input"):
            return
        if tail is None or tail.size == 0:
            tail = np.zeros(self.IN_SR, dtype=np.float32)
        elif tail.size < self.IN_SR:
            tail = np.concatenate([tail, np.zeros(self.IN_SR - tail.size, dtype=tail.dtype)])
        self._model.streaming_prefill(
            session_id=self._sid_token(sid),
            msgs=[{"role": "user", "content": [tail]}],
            omni_mode=False,
            is_last_chunk=True,
        )

    def _streaming_generate(self, sid: str):
        sc = self.stream_cfg
        # 注意：不要再 reset_session —— 同一 session_id 的 prefill 上下文要保留到
        # 本轮生成当中。duplex 轮与轮之间才需要由上层决定是否重置（默认不清）。
        return self._model.streaming_generate(
            session_id=self._sid_token(sid),
            generate_audio=sc.get("generate_audio", True),
            use_tts_template=sc.get("use_tts_template", True),
            enable_thinking=sc.get("enable_thinking", False),
            do_sample=sc.get("do_sample", True),
            max_new_tokens=sc.get("max_new_tokens", 512),
            length_penalty=sc.get("length_penalty", 1.1),
        )

    # ---------------- 护栏澄清快路径 ----------------
    def speak_preview(self, text: str, sid: Optional[str] = None) -> StreamChunk:
        """护栏触发澄清/停止时，让模型把指定文本逐字念出来。

        用官方 streaming_generate 的 teacher_forcing 分支：把 text 的 token
        逐个喂入 LLM，取对应 hidden states 构造 TTS 条件，保证输出音频与
        文本精确一致（而非模型自由发挥）。TTS 延续 tts_last_turn_tokens，
        澄清语音与正常回答音色连续。

        - 传 sid：在主会话上收尾用户轮后做 teacher forcing，护栏话术作为
          真实 assistant 轮进入 KV 上下文——下一轮模型知道自己问过澄清。
        - 不传 sid：开独立 preview 会话（1s 静音占位用户轮），不依赖主会话。
        """
        if self._model is None:
            self.load_model()
        sc = self.stream_cfg
        if sid is not None:
            st = self._state.get(sid)
            if st is None:
                raise KeyError(f"unknown duplex session: {sid}")
            self._ensure_system_prefill(sid)
            self._close_user_turn(sid)   # 收尾用户轮，接续同一 KV cache
            token = self._sid_token(sid)
        else:
            import numpy as np
            token = f"preview_{uuid.uuid4().hex[:8]}"
            silent = np.zeros(self.IN_SR, dtype=np.float32)
            self._model.streaming_prefill(
                session_id=token,
                msgs=[{"role": "user", "content": [silent]}],
                omni_mode=False,
                is_last_chunk=True,
            )
        audio_all, text_seen = b"", ""
        for wav, tchunk in self._model.streaming_generate(
            session_id=token,
            generate_audio=True,
            use_tts_template=sc.get("use_tts_template", True),
            enable_thinking=False,
            do_sample=False,
            max_new_tokens=256,
            teacher_forcing=True,
            teacher_forcing_text=text,
        ):
            if tchunk:
                text_seen += tchunk
            if wav is not None:
                audio_all += self._float_to_pcm16(wav)
        if not audio_all:
            raise RuntimeError("speak_preview: teacher forcing 未产出任何音频")
        if sid is not None:
            st = self._state[sid]
            st["pending"], st["samples"], st["had_input"] = [], 0, False
        return StreamChunk(
            kind="bot_audio", audio=audio_all, text=text,
            meta={"mode": "teacher_forcing", "forced_text": text,
                  "model_echo": text_seen},
        )


# ----------------------------------------------------------------------
# MiniCPM-o duplex_server 模式（P2+）：连接官方 wss 全双工服务
# 协议（见官方 Duplex Mode 文档）：
#   connect -> queued(queue_done) -> send prepare{system_prompt,config} -> prepared
#   循环：每 chunk 发 audio_chunk -> result{is_listen, text, audio_data, end_of_turn}
#   用户抢话时发 set_break 打断；结束发 stop。
# 真正的"边听边说 + 自主听/说 + 可打断"，模型由远端服务承载，本地无需加载权重。
# ----------------------------------------------------------------------
class MiniCPMODuplexServerBackend(DuplexBackend):
    """通过官方 wss 服务实现真·全双工（stream_style='duplex_server'，P2+）。

    配置键与 configs/base.yaml 的 duplex.duplex_server 对齐：
    ws_url / session_prefix / chunk_ms / force_listen_count /
    max_new_speak_tokens_per_chunk / listen_prob_scale / ls_mode /
    temperature / top_p / top_k / ref_audio_path。
    """
    backend_name = "minicpm_o"
    IN_SR = 16000
    OUT_SR = 24000

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dup = cfg.get("duplex", {})
        ds = dup.get("duplex_server", {}) or {}
        self.ws_url_tpl = ds.get("ws_url", "ws://127.0.0.1:8006/ws/duplex/{session_id}")
        self.session_prefix = ds.get("session_prefix", "adx_")
        self.chunk_ms = int(ds.get("chunk_ms", 1000))
        self.force_listen_count = int(ds.get("force_listen_count", 3))
        # 等 result 的超时：不能复用 first_token_budget_s（0.8s 会秒超时）
        self.timeout_s = float(ds.get("timeout_s", 30))
        # 构造 duplex config 时传给服务端，正是官方 DuplexConfig 的字段
        self.duplex_config = {
            "chunk_ms": self.chunk_ms,
            "force_listen_count": self.force_listen_count,
            "max_new_speak_tokens_per_chunk": ds.get("max_new_speak_tokens_per_chunk", 20),
            "listen_prob_scale": ds.get("listen_prob_scale", 1.0),
            "ls_mode": ds.get("ls_mode", "explicit"),
            "temperature": ds.get("temperature", 0.7),
            "top_p": ds.get("top_p", 0.8),
            "top_k": ds.get("top_k", 20),
            "ref_audio_path": ds.get("ref_audio_path", "") or None,
        }
        self.system_prompt = ds.get("system_prompt",
                                    "You are a helpful assistant.")
        self._ws: dict[str, object] = {}          # sid -> websocket
        self._inbox: dict[str, "queue.Queue"] = {}  # sid -> result 队列
        self._prepared: set[str] = set()

    def _url(self, sid: str) -> str:
        return self.ws_url_tpl.format(session_id=f"{self.session_prefix}{sid}")

    def load_model(self):
        # duplex_server 模式下模型跑在远端服务，本地无需加载权重
        logger.info("duplex_server mode: 模型由远端 wss 服务承载，本地不加载权重。")
        return None

    def register(self, sid: str) -> DuplexSession:
        import queue
        s = super().register(sid)
        self._inbox[sid] = queue.Queue()
        return s

    def close(self, sid: str):
        ws = self._ws.pop(sid, None)
        if ws is not None:
            try:
                ws.send('{"type":"stop"}')
                ws.close()
            except Exception:  # pragma: no cover - 网络关闭容错
                pass
        self._inbox.pop(sid, None)
        self._prepared.discard(sid)
        super().close(sid)

    def push_user_audio(self, sid: str, chunk: StreamChunk) -> None:
        import websocket  # websocket-client
        ws = self._ws.get(sid)
        if ws is None:
            ws = self._connect(sid)
        self._ensure_prepared(sid, ws)
        # 每次推理默认 1s=16000 采样点；这里按后端 chunk 约定传 pcm 字节
        payload = {
            "type": "audio_chunk",
            "audio_data": _pcm16_b64(chunk.audio) if chunk.audio else None,
        }
        ws.send(json.dumps(payload))

    def iter_bot_output(self, sid: str) -> Iterator[StreamChunk]:
        import queue
        inbox = self._inbox[sid]
        while True:
            try:
                msg = inbox.get(timeout=self.timeout_s)
            except queue.Empty:
                logger.warning("duplex_server: 等待结果超时")
                return
            rtype = msg.get("type")
            if rtype != "result":
                continue   # 忽略 queue_update/queued 等副消息
            is_listen = bool(msg.get("is_listen"))
            text = msg.get("text") or ""
            end = bool(msg.get("end_of_turn"))
            if not is_listen:      # 模型在说话
                yield StreamChunk(kind="bot_text", text=text)
                audio = msg.get("audio_data")
                if audio:
                    yield StreamChunk(kind="bot_audio",
                                      audio=_b64_to_pcm16(audio))
            if end:
                return

    def interrupt(self, sid: str):
        """用户抢话打断：通知模型停下当前输出转向聆听。"""
        ws = self._ws.get(sid)
        if ws is not None:
            ws.send(json.dumps({"type": "set_break"}))

    def speak_preview(self, text: str, sid: Optional[str] = None) -> StreamChunk:
        """wss 模式下无法本地合成任意文本语音；澄清文案以事件形式交给上层展示。"""
        return StreamChunk(kind="event", text=text,
                           meta={"note": "duplex_server 澄清由上层/UI 播报"})

    # ---------------- 内部连接逻辑 ----------------
    def _connect(self, sid: str):
        import threading
        import websocket  # websocket-client
        url = self._url(sid)
        logger.info(f"connecting duplex_server: {url}")
        ws = websocket.create_connection(url, timeout=self.timeout_s)
        self._ws[sid] = ws  # 重新赋值，避免读线程闭包旧引用
        # 消费端：把服务端消息投递到 inbox 队列
        def _pump():
            try:
                while True:
                    raw = ws.recv()
                    if not raw:
                        break
                    msg = json.loads(raw)
                    if msg.get("type") in ("queued", "queue_done", "queue_update"):
                        continue
                    self._inbox[sid].put(msg)
            except Exception:
                self.close(sid)
        threading.Thread(target=_pump, daemon=True, name=f"duplex-{sid}").start()
        return self._ws[sid]

    def _ensure_prepared(self, sid: str, ws):
        if sid in self._prepared:
            return
        prepare = {
            "type": "prepare",
            "system_prompt": self.system_prompt,
            "config": {
                "sample_rate_in": self.IN_SR,
                "sample_rate_out": self.OUT_SR,
                **self.duplex_config,
            },
        }
        ws.send(json.dumps(prepare))
        self._prepared.add(sid)


def _pcm16_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _b64_to_pcm16(b64: str) -> bytes:
    import base64
    return base64.b64decode(b64)


# ----------------------------------------------------------------------
# 缓冲管理器：实现"边听边生成不互相阻塞"的流缓冲
# ----------------------------------------------------------------------
class DuplexBuffer:
    """并听并说缓冲：user 输入流与 bot 输出流并存，提供打断检测。"""

    def __init__(self, cfg: dict):
        d = cfg.get("duplex", {})
        self.chunk_s = d.get("chunk_seconds", 0.5)
        self.overlap_thr = d.get("barge_in_overlap_s", 0.15)
        self.user_audio: list = []
        self.bot_audio: list = []
        self._last_user_ts = 0.0
        self._last_bot_ts = 0.0

    def feed_user(self, chunk: StreamChunk):
        self.user_audio.append(chunk)
        self._last_user_ts = chunk.ts
        return self._detect_barge_in()

    def feed_bot(self, chunk: StreamChunk):
        self.bot_audio.append(chunk)
        self._last_bot_ts = chunk.ts

    def _detect_barge_in(self) -> bool:
        """当用户音频与 bot 输出时间上重叠超过阈值，判定为"用户抢话"。"""
        if not self.user_audio or not self.bot_audio:
            return False
        overlap = min(self._last_user_ts, self._last_bot_ts) - max(
            self.user_audio[-1].ts, self.bot_audio[-1].ts
        )
        return overlap > self.overlap_thr

    def user_is_active(self) -> bool:
        return bool(self.user_audio) and self._last_user_ts >= self._last_bot_ts

    def reset_cycle(self):
        self.bot_audio = []
        self.user_audio = []
        self._last_bot_ts = 0.0


def build_backend(cfg: dict) -> DuplexBackend:
    """工厂：依 config backend 与 duplex.stream_style 构建后端。

    - minicpm_o + streaming_generate -> MiniCPMOBackend（P0/P1, 本地直接接线）
    - minicpm_o + duplex_server     -> MiniCPMODuplexServerBackend（P2+, 官方 wss 服务）
    """
    name = cfg.get("backend", "minicpm_o")
    if name == "minicpm_o":
        style = cfg.get("duplex", {}).get("stream_style", "streaming_generate")
        if style == "duplex_server":
            return MiniCPMODuplexServerBackend(cfg)
        return MiniCPMOBackend(cfg)
    table = {
        "nemotron_voicechat": _need_stub("nemotron_voicechat"),
        "moshi": _need_stub("moshi"),
    }
    cls = table.get(name)
    if cls is None:
        raise NotImplementedError(f"unsupported backend: {name}")
    return cls(cfg)


def _need_stub(name: str):
    def _make(cfg):
        logger.warning(
            f"backend '{name}' 尚未实现加载封装，请为它补一份 DuplexBackend 实现。"
            "本项目当前聚焦 MiniCPM-o / VoiceChat；两者流式接口差异不大。"
        )
        return _StubBackend(name, cfg)
    return _make


class _StubBackend(DuplexBackend):
    backend_name = "stub"

    def __init__(self, name, cfg):
        super().__init__(cfg)
        self.backend_name = name

    def load_model(self):
        raise NotImplementedError(f"{self.backend_name} backend stub")

    def push_user_audio(self, sid, chunk): ...
    def iter_bot_output(self, sid): ...
    def speak_preview(self, text, sid=None): ...