"""本机可跑的测试：无需 GPU，验证护栏与指标逻辑正确。

运行：python -m pytest tests/ -q  （或 AutoDL 上机前先本机跑一遍）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYS = sys.path[:]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from tfd.gate.trust_gate import TrustGate, EXECUTE, CLARIFY, STOP
from tfd.gate.features import _normalize_entropy, _scale_logprob
from tfd.eval.metrics import misuse_rate, content_bleu, first_token_latency
from tfd.rl.rewards import total_reward, grpo_advantage, TurnSample
from tfd.utils.config import load_config


def _gate():
    return TrustGate(load_config("gate"))


def test_low_risk_clear_executes():
    g = _gate()
    d = g.on_frame(text="今天天气怎么样", logprob=-0.05, asr_confidence=0.9)
    assert d.action == EXECUTE, d


def test_high_risk_stops():
    g = _gate()
    d = g.on_frame(text="现在就右转然后变道", logprob=-0.5, asr_confidence=0.6)
    assert d.action == STOP, d


def test_ambiguous_clarifies_after_frames():
    g = _gate()
    d = None
    for _ in range(4):
        d = g.on_frame(text="帮我开一下那个", logprob=-0.6, asr_confidence=0.45)
    final = g.finalize()
    assert final.action == CLARIFY, final


def test_entropy_normalized():
    assert 0.0 <= _normalize_entropy([0.9, 0.1]) <= 1.0
    assert _normalize_entropy([0.25, 0.25, 0.25, 0.25]) == 1.0


def test_logprob_mapping():
    assert _scale_logprob(-0.05) > _scale_logprob(-0.6)


def test_misuse_rate():
    assert misuse_rate(1, 4) == 0.25


def test_content_bleu_punishes_hallucination():
    ref = "帮我把明早七点的闹钟设一下"
    good = "设置了明早七点的闹钟"
    bad = "忘记你说什么了"
    assert content_bleu(ref, good) > content_bleu(ref, bad)


def test_first_token_latency():
    assert abs(first_token_latency(1.0, 1.42) - 0.42) < 1e-6


def test_reward_penalizes_wrong_gate_action():
    weights = load_config("rl")["rl"]["reward_weights"]
    good = TurnSample(0.8, 600, 0.9, 0.2,
                      meta={"gate_expected": "stop", "model_action": "stop"})
    bad = TurnSample(0.8, 600, 0.9, 0.2,
                     meta={"gate_expected": "stop", "model_action": "execute"})
    assert total_reward(good, weights) > total_reward(bad, weights)


def test_grpo_advantage_centered():
    r = [1.0, 2.0, 3.0]
    advs = grpo_advantage(r)
    assert abs(sum(advs)) < 1e-6  # 标准差归一化后近似 sum=0


# ---------------- 方向C：风险锁存 + 不可逆 bonus 修复 ----------------

def test_risk_latch_and_reset():
    g = _gate()
    g.on_frame(text="今天天气怎么样", logprob=-0.05, asr_confidence=0.9)
    assert not g.risk_seen and g.first_risk_frame is None
    g.on_frame(text="帮我把相册里的视频删了", logprob=-0.05, asr_confidence=0.9)
    assert g.risk_seen and g.first_risk_frame == 1
    g.reset()
    assert not g.risk_seen and g.first_risk_frame is None and g._frame_idx == 0


def test_clear_voice_irreversible_never_executes():
    """清晰语音的不可逆指令必须二次确认（irreversible bonus 修复的回归）。"""
    g = _gate()
    d = None
    for _ in range(4):
        d = g.on_frame(text="帮我把相册里的视频删了",
                       logprob=-0.05, asr_confidence=0.9)
    assert d.action == CLARIFY, d      # risk 0.4 >= floor 0.4 -> 不许直接执行
    assert g.finalize().action != EXECUTE


def test_risk_estimator_bonuses():
    from tfd.gate.features import RiskEstimator
    est = RiskEstimator(load_config("gate"))
    assert est.estimate("帮我删除这个文件", None) == 0.4    # 不可逆类
    assert est.estimate("现在就变道", None) == 0.5          # 物理类
    assert est.estimate("删除之后马上变道", None) == 0.5    # 取 max
    assert est.estimate("给我讲个笑话", None) == 0.0        # 良性
    assert 0.4 <= est.estimate("删除", None)                # 不可逆不低于确认线


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))