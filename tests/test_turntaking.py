"""方向A/B 新增模块单元测试：turn-taking 决策器 + barge-in 机制 + 63 脚本接线。

运行：python -m pytest tests/ -q（无 GPU 依赖；LearnedMLP 用 CPU 小样本）
"""
import importlib
import json
import sys

import numpy as np

from tfd.turntaking.synth import synth_dataset, frame_labels, HOLD, TAKE, BACKCHANNEL
from tfd.turntaking.features import (
    FrameFeaturizer, FEATURE_DIM, featurize_utterance,
)
from tfd.turntaking.predictor import (
    RulePauseStrategy, TurnStrategy, TrainableMLP, evaluate_strategy,
)
from tfd.turntaking.bargein import (
    EnergyVAD, InterruptiblePlayback, make_mic_stream, frame_db,
)


# ---------------- 方向A：合成数据与流式特征 ----------------

def test_synth_dataset_reproducible():
    a = synth_dataset(5, seed=7)
    b = synth_dataset(5, seed=7)
    assert [u.segs for u in a] == [u.segs for u in b]
    assert a[0].final_pause_start > 0


def test_frame_labels_semantics():
    for u in synth_dataset(10, seed=1):
        labels = frame_labels(u)
        for i, lb in enumerate(labels):
            if lb == TAKE:                       # take 只在结尾停顿 + delay 之后
                assert (i + 1) * 0.1 > u.final_pause_start + 0.2
            if lb == BACKCHANNEL:                # backchannel 只在犹豫窗口内
                assert any(s + 0.2 < (i + 1) * 0.1 <= e
                           for s, e in u.filler_windows)
        assert labels[-1] == TAKE                # 结尾停顿 >=0.7s，末帧必为 take


def test_featurize_streaming_shape():
    utt = synth_dataset(1, seed=3)[0]
    X, dbs = featurize_utterance(utt)
    n_frames = int(np.ceil(utt.total_s / 0.1))
    assert X.shape == (n_frames, FEATURE_DIM)
    assert np.isfinite(X).all()


def test_featurizer_no_future_leak():
    """前缀帧特征只依赖已见音频：与全量喂入的对应帧逐帧一致。"""
    audio = synth_dataset(1, seed=5)[0].to_audio()
    fn = 1600
    n = 20
    full = FrameFeaturizer()
    prefix = FrameFeaturizer()
    f_full = [full.feed(audio[i * fn:(i + 1) * fn]) for i in range(n)]
    f_pref = [prefix.feed(audio[i * fn:(i + 1) * fn]) for i in range(n)]
    assert np.allclose(f_full, f_pref)


# ---------------- 方向A：策略与事件级评测 ----------------

def test_rule_pause_threshold():
    s = RulePauseStrategy(threshold_ms=500)
    assert s.feed([0.0, 1.0] + [0.0] * 8) == TAKE      # pause_len 特征 1.0 -> 1.5s
    assert s.feed([0.0, 0.1] + [0.0] * 8) == HOLD      # 0.15s 不够


def test_trainable_mlp_learns_pause():
    """小样本可分任务上 MLP 帧级精度应显著高于随机（1/3）。"""
    X, y = [], []
    for u in synth_dataset(30, seed=11):
        feats, _ = featurize_utterance(u)
        X.extend(feats.tolist())
        y.extend(frame_labels(u))
    X = np.asarray(X, np.float32)
    mlp = TrainableMLP(hidden=16, lr=0.05, seed=0)
    mlp.fit(X, y, epochs=60, verbose_every=0)
    assert mlp.accuracy(X, y) > 0.6


class _TakeAt(TurnStrategy):
    """在指定帧 take 的假策略，验证事件级口径的数学。"""
    name = "take_at"

    def __init__(self, take_frame):
        self.take_frame = take_frame
        self.i = 0

    def feed(self, feats):
        self.i += 1
        return TAKE if (self.i - 1) == self.take_frame else HOLD

    def reset(self):
        self.i = 0


def test_evaluate_event_semantics():
    n = 30
    feats = np.zeros((n, FEATURE_DIM), np.float32)
    labels = [HOLD] * n
    # 抢话：take 落在结尾停顿开始之前
    ev = evaluate_strategy(_TakeAt(5), feats, labels, final_pause_start_ms=1200)
    assert ev.false_takeover and ev.take_at_ms == 600
    # 正常接话：延迟 = take 时刻 - 结尾停顿起点
    labels[25:] = [TAKE] * 5
    ev = evaluate_strategy(_TakeAt(15), feats, labels, final_pause_start_ms=1200)
    assert not ev.false_takeover and ev.take_latency_ms == 400 and not ev.miss
    # 漏接：从不 take 且话语有 take 标签
    ev = evaluate_strategy(_TakeAt(None), feats, labels, final_pause_start_ms=1200)
    assert ev.miss and ev.take_at_ms is None


# ---------------- 方向B：barge-in 检测与可中断播放 ----------------

def test_energy_vad_sustain_and_reset():
    vad = EnergyVAD(threshold_db=-38.0, frame_ms=100, sustain_frames=3)
    loud = np.full(1600, 0.3, np.float32)      # ~ -10dB
    quiet = np.full(1600, 0.001, np.float32)   # ~ -60dB
    assert not vad.feed_db(frame_db(loud))
    assert not vad.feed_db(frame_db(loud))
    assert vad.feed_db(frame_db(loud))          # 第 3 帧触发
    assert not vad.feed_db(frame_db(loud))      # 只触发一次
    vad.reset()
    assert not vad.feed_db(frame_db(loud))
    assert not vad.feed_db(frame_db(quiet))     # 静音打断 run，重新计数
    assert not vad.feed_db(frame_db(loud))
    assert not vad.feed_db(frame_db(loud))
    assert vad.feed_db(frame_db(loud))


def test_interruptible_playback_latency():
    """插话@800ms, chunk=200ms：检测 300ms（3 帧确认），停在 chunk 边界。"""
    vad = EnergyVAD(threshold_db=-38.0, frame_ms=100, sustain_frames=3)
    pb = InterruptiblePlayback(vad, chunk_ms=200)
    tts = iter([b"\x01\x00" * 4800] * 50)      # 50 x 200ms
    mic = make_mic_stream(bot_len_ms=10000, bargein_at_ms=800, rng_seed=0)
    res = pb.play(tts, mic)
    assert res.interrupted
    assert res.bargein_at_ms == 800.0
    assert res.detection_latency_ms == 300.0
    assert res.stopped_at_ms == 1200.0          # VAD 触发@1100 -> chunk 边界 1200
    assert res.stop_latency_ms == 100.0


def test_playback_without_bargein():
    """插话落在播报之后：播完不触发打断。"""
    vad = EnergyVAD(threshold_db=-38.0, frame_ms=100, sustain_frames=3)
    pb = InterruptiblePlayback(vad, chunk_ms=200)
    tts = iter([b"\x01\x00" * 4800] * 10)
    mic = make_mic_stream(bot_len_ms=10000, bargein_at_ms=100000, rng_seed=0)
    res = pb.play(tts, mic)
    assert not res.interrupted
    assert res.played_ms == 2000.0


# ---------------- 63 脚本：离线接线回归 ----------------

def _mod63():
    return importlib.import_module("63_duplex_bargein")


def test_bigram_overlap():
    m = _mod63()
    assert m.bigram_overlap("删除相册视频", "删除相册视频") == 1.0
    assert m.bigram_overlap("删除相册视频", "今天天气很好") == 0.0
    assert 0.0 < m.bigram_overlap("回答问题设置闹钟", "我可以回答问题") < 1.0


def test_63_offline_wiring():
    """63 离线全流程（FakeModel）：截断 -> 新轮 -> 探针 -> 报告落盘。"""
    from bootstrap import ROOT
    old_argv = sys.argv
    sys.argv = ["63_duplex_bargein.py", "--offline"]
    try:
        assert _mod63().main() == 0
    finally:
        sys.argv = old_argv
    r = json.loads((ROOT / "outputs" / "duplex_session" / "bargein_report.json")
                   .read_text(encoding="utf-8"))
    assert r["mode"] == "offline"
    assert r["turn1_cut"]["interrupted"] is True
    assert r["turn2_bargein"]["ttft_ms"] is not None
    assert r["context_preserved_heuristic"] is True
