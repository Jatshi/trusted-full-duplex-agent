"""backends 接线单元测试：用 FakeModel 验证分块/收轮/系统轮逻辑（无需 GPU/torch）。"""
import numpy as np
import pytest

from tfd.base.backends import (
    MiniCPMOBackend, MiniCPMODuplexServerBackend, build_backend, StreamChunk,
)


def make_cfg(stream_style="streaming_generate"):
    return {
        "backend": "minicpm_o",
        "paths": {"weights_dir": "weights/"},
        "model": {"minicpm_o": {
            "repo_or_path": "openbmb/MiniCPM-o-4_5",
            "dtype": "bf16", "load_from_local": False, "device_map": "auto",
        }},
        "duplex": {
            "stream_style": stream_style,
            "chunk_seconds": 0.5,
            "streaming_generate": {
                "session_id_prefix": "tfd_", "generate_audio": True,
                "use_tts_template": True, "max_new_tokens": 64,
            },
            "duplex_server": {"ws_url": "ws://h:1/ws/duplex/{session_id}"},
        },
    }


class FakeModel:
    """记录 streaming_prefill 调用；streaming_generate 产出一组 (wav, text)。"""

    def __init__(self):
        self.prefills = []          # (session_id, n_samples, is_last_chunk, role)
        self.generate_calls = 0
        self.last_gen_kwargs = None

    def streaming_prefill(self, session_id, msgs, omni_mode, is_last_chunk):
        content = msgs[0]["content"]
        audio = content[0] if content and not isinstance(content[0], str) else None
        self.prefills.append((session_id, None if audio is None else len(audio),
                              is_last_chunk, msgs[0]["role"]))

    def streaming_generate(self, **kw):
        self.generate_calls += 1
        self.last_gen_kwargs = kw
        wav = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
        yield wav, "你好"
        yield None, "，世界"


def make_backend():
    b = MiniCPMOBackend(make_cfg())
    b._model = FakeModel()
    return b


def pcm_of(seconds: float, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2").tobytes()


def test_chunking_and_turn_close():
    """1.5s 音频 -> 系统轮1次 + 1s 全块 prefill + 残块补零收尾(is_last=True)。"""
    b = make_backend()
    sid = "s1"
    b.register(sid)
    b.push_user_audio(sid, StreamChunk(kind="user_audio", audio=pcm_of(1.5)))
    chunks = list(b.iter_bot_output(sid))
    m = b._model
    roles = [p[3] for p in m.prefills]
    assert roles[0] == "system"                       # 系统轮最先
    assert roles.count("system") == 1                 # 只一次
    user = [p for p in m.prefills if p[3] == "user"]
    assert len(user) == 2                             # 1s 全块 + 收尾块
    assert user[0][2] is False and user[0][1] == 16000
    assert user[1][2] is True and user[1][1] == 16000  # 残块补零到 1s 且标记 last
    assert m.generate_calls == 1
    kinds = [c.kind for c in chunks]
    assert "bot_text" in kinds and "bot_audio" in kinds


def test_exact_seconds_still_closes_turn():
    """整 2s 音频（无残块）也必须有一块 is_last=True 收尾。"""
    b = make_backend()
    sid = "s2"
    b.register(sid)
    b.push_user_audio(sid, StreamChunk(kind="user_audio", audio=pcm_of(2.0)))
    list(b.iter_bot_output(sid))
    user = [p for p in b._model.prefills if p[3] == "user"]
    assert len(user) == 3                             # 2 全块 + 1 补零收尾
    assert user[-1][2] is True and user[-1][1] == 16000


def test_no_input_no_prefill():
    """没推过音频时 iter 不应发任何 user prefill（不对静音应答）。"""
    b = make_backend()
    sid = "s3"
    b.register(sid)
    list(b.iter_bot_output(sid))
    assert [p for p in b._model.prefills if p[3] == "user"] == []


def test_partial_chunks_accumulate():
    """0.5s chunk 喂两次才凑满 1s：prefill 只发生一次。"""
    b = make_backend()
    sid = "s4"
    b.register(sid)
    b.push_user_audio(sid, StreamChunk(kind="user_audio", audio=pcm_of(0.5)))
    assert [p for p in b._model.prefills if p[3] == "user"] == []
    b.push_user_audio(sid, StreamChunk(kind="user_audio", audio=pcm_of(0.5)))
    user = [p for p in b._model.prefills if p[3] == "user"]
    assert len(user) == 1 and user[0][1] == 16000 and user[0][2] is False


def test_build_backend_routing():
    assert isinstance(build_backend(make_cfg()), MiniCPMOBackend)
    assert isinstance(build_backend(make_cfg("duplex_server")),
                      MiniCPMODuplexServerBackend)


def test_duplex_server_url_from_yaml():
    b = MiniCPMODuplexServerBackend(make_cfg("duplex_server"))
    assert b._url("abc") == "ws://h:1/ws/duplex/adx_abc"
    assert b.timeout_s >= 30            # 不复用 0.8s 的首 token 预算


def test_pcm_roundtrip():
    pcm = pcm_of(0.25)
    f = MiniCPMOBackend._pcm16_to_float(pcm)
    back = MiniCPMOBackend._float_to_pcm16(f)
    assert len(back) == len(pcm)
    assert np.abs(np.frombuffer(back, "<i2").astype(np.int32)
                  - np.frombuffer(pcm, "<i2").astype(np.int32)).max() <= 1


def test_unknown_session_raises():
    b = make_backend()
    with pytest.raises(KeyError):
        b.push_user_audio("nope", StreamChunk(kind="user_audio", audio=b"\x00\x00"))


def test_speak_preview_on_main_session():
    """护栏拦截快路径：主会话上 teacher_forcing，收尾用户轮 + 话术进入上下文。"""
    b = make_backend()
    sid = "s5"
    b.register(sid)
    b.push_user_audio(sid, StreamChunk(kind="user_audio", audio=pcm_of(0.5)))
    pv = b.speak_preview("请再具体说一次", sid=sid)
    m = b._model
    kw = m.last_gen_kwargs
    assert kw["teacher_forcing"] is True
    assert kw["teacher_forcing_text"] == "请再具体说一次"
    assert kw["generate_audio"] is True
    # 用户轮在主会话 token 上收尾（系统轮 + 残块补零 is_last=True）
    user = [p for p in m.prefills if p[3] == "user"]
    assert len(user) == 1 and user[0][0] == "tfd_s5" and user[0][2] is True
    # 返回完整音频 + 话术元信息；会话状态复位
    assert pv.kind == "bot_audio" and len(pv.audio) > 0
    assert pv.meta["forced_text"] == "请再具体说一次"
    st = b._state[sid]
    assert st["pending"] == [] and st["had_input"] is False


def test_speak_preview_isolated_session():
    """不传 sid：独立 preview 会话（1s 静音用户轮），同样走 teacher_forcing。"""
    b = make_backend()
    pv = b.speak_preview("该操作有风险，已停止")
    m = b._model
    kw = m.last_gen_kwargs
    assert kw["teacher_forcing"] is True
    assert kw["teacher_forcing_text"] == "该操作有风险，已停止"
    user = [p for p in m.prefills if p[3] == "user"]
    assert len(user) == 1 and user[0][0].startswith("preview_")
    assert user[0][1] == 16000 and user[0][2] is True
    assert len(pv.audio) > 0


def test_grpo_trainer_smoke():
    """有 torch 才跑：验证一步 GRPO 的梯度真正回传（AutoDL 上必跑）。"""
    torch = pytest.importorskip("torch")
    from tfd.rl.grpo_trainer import GRPOTrainer

    class TinyLM(torch.nn.Module):
        def __init__(self, vocab=32):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, 16)
            self.head = torch.nn.Linear(16, vocab)

        def forward(self, input_ids):
            return type("O", (), {"logits": self.head(self.emb(input_ids))})()

        def generate(self, input_ids, **kw):
            n = kw.get("num_return_sequences", 1)
            new = torch.randint(0, 32, (n, 5))
            return torch.cat([input_ids.expand(n, -1), new], dim=1)

    class FakeTok:
        pad_token_id = 0
        def batch_decode(self, ids, **kw):
            # 第 j 条回复截断到不同长度 -> BLEU 各异 -> 组内优势非零
            ref = "示例回复"
            return [ref[: max(1, len(ref) - j)] for j in range(ids.shape[0])]

    cfg = {"rl": {"kl_scale": 0.0, "group_size": 4, "reward_weights": {}},
           "training": {"lr": 1e-3}}
    m = TinyLM()
    t = GRPOTrainer(cfg, m, None, m, tokenizer=FakeTok())
    prompt = torch.randint(0, 32, (1, 3))
    loss = t.train_step_on(prompt, ["示例回复"], "execute")
    assert isinstance(loss, float)
    # 关键断言：参数真的被更新过（梯度链没断、组内优势非零）
    assert m.emb.weight.grad is not None
    assert m.emb.weight.grad.abs().sum() > 0


def test_seq_logprobs_differentiable():
    """有 torch 才跑：_seq_logprobs 输出可微，backward 能回填 embedding 梯度。"""
    torch = pytest.importorskip("torch")
    from tfd.rl.grpo_trainer import _seq_logprobs

    emb = torch.nn.Embedding(32, 8)
    head = torch.nn.Linear(8, 32)

    class M(torch.nn.Module):
        def forward(self, input_ids):
            return type("O", (), {"logits": head(emb(input_ids))})()

    m = M()
    ids = torch.randint(0, 32, (2, 6))
    mask = torch.zeros(2, 5)
    mask[:, 2:] = 1.0
    lp = _seq_logprobs(m, ids, mask)
    assert lp.shape == (2,)
    assert lp.requires_grad
    (-lp.mean()).backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad.abs().sum() > 0