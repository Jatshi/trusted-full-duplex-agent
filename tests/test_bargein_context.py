"""barge-in 上下文训练闭环单元测试：33 数据 + context_recall reward + 32 接线。

运行：python -m pytest tests/test_bargein_context.py -q
"""
import importlib
import json

from tfd.eval.metrics import bigram_overlap
from tfd.rl.grpo_trainer import bargein_context_reward, grpo_generation_reward


# ---------------- reward：排序正确性 ----------------

IC = "你好，我是车载语音助手小智。我可以回答问题、设置闹钟"
W = {"context_recall": 1.0, "content_consistency": 0.5}


def test_context_reward_ranks_recall_above_hallucination():
    """准确复述 > 编造未播内容 > 无关回答。"""
    good = "您打断之前，我说到我可以回答问题、设置闹钟。"
    hallucinated = "我刚才说到我可以查询天气、控制智能家居，还能帮你订咖啡。"
    unrelated = "今天天气不错，适合出行。"
    r_good = bargein_context_reward(good, good, IC, W)
    r_hallu = bargein_context_reward(hallucinated, good, IC, W)
    r_unrel = bargein_context_reward(unrelated, good, IC, W)
    assert r_good > r_hallu
    assert r_good > r_unrel
    assert r_unrel < 0.15


def test_context_reward_perfect_recap_is_high():
    """逐字复述被截断内容应得高分（bigram=1 主导）。"""
    perfect = IC
    r = bargein_context_reward(perfect, perfect, IC, W)
    assert r > 0.6


def test_bigram_overlap_shared_metric():
    """63 与 RL 共用同一指标实现（口径一致是闭环的前提）。"""
    assert bigram_overlap("abc", "abc") == 1.0
    assert bigram_overlap("", "") == 0.0
    assert bigram_overlap("设置闹钟提醒", "设置闹钟提醒") > 0.99


# ---------------- 33 数据生成 ----------------

def _gen33(tmp_path):
    m = importlib.import_module("33_prep_bargein_rl_data")
    return m.gen_bargein_context(tmp_path)


def test_prep33_generates_valid_rows(tmp_path):
    p = _gen33(tmp_path)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]
    assert len(rows) == 60
    for r in rows:
        assert r["family"] == "barge_in_context"
        assert len(r["context"]) == 3
        assert r["context"][1]["role"] == "assistant"   # 悬空被截断轮
        assert not r["context"][1]["content"].endswith(("。", "！", "？"))  # 句中悬空
        assert r["context"][2]["role"] == "user"        # 抢话轮
        assert r["interrupted_content"] == r["context"][1]["content"]
        assert r["interrupted_content"] in r["response"]  # 参考复述包含已播内容
    # 确定性：同参数两次生成结果一致
    p2 = _gen33(tmp_path)
    assert p2.read_text(encoding="utf-8") == p.read_text(encoding="utf-8")


def test_prep33_rows_scoreable_by_context_reward():
    """生成的行可直接被 bargein_context_reward 打分（字段对齐）。"""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = _gen33(Path(d))
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()[:5]]
    for r in rows:
        good = r["interrupted_content"]
        score = bargein_context_reward(good, r["response"],
                                       r["interrupted_content"], W)
        assert score > 0.5


# ---------------- 32 接线：多轮 prompt 与族感知 reward ----------------

class _StubTok:
    """记录 apply_chat_template 收到的消息，返回假 ids 张量。"""

    def __init__(self):
        self.seen = None

    def apply_chat_template(self, msgs, **kw):
        self.seen = msgs
        import torch
        return torch.tensor([[1, 2, 3]])


def test_build_prompt_includes_context_turns():
    m = importlib.import_module("32_run_grpo")
    tok = _StubTok()
    row = {
        "system": "SYS", "instruction": "Q",
        "context": [
            {"role": "user", "content": "请介绍一下你自己"},
            {"role": "assistant", "content": "你好，我是"},
            {"role": "user", "content": "等一下，先停一下。"},
        ],
    }
    m._build_prompt(tok, row, device="cpu")
    roles = [msg["role"] for msg in tok.seen]
    assert roles == ["system", "user", "assistant", "user", "user"]
    assert tok.seen[2]["content"] == "你好，我是"          # 悬空轮在 prompt 里
    assert tok.seen[-1]["content"] == "Q"                   # 探针是最后 user 轮


def test_build_prompt_plain_row_compat():
    """旧行为兼容：无 context 的行（31 数据）仍构造 system+user 两轮。"""
    m = importlib.import_module("32_run_grpo")
    tok = _StubTok()
    m._build_prompt(tok, {"instruction": "今天天气怎么样"}, device="cpu")
    assert [msg["role"] for msg in tok.seen] == ["system", "user"]
    m._build_prompt(tok, "今天天气怎么样", device="cpu")     # 裸字符串也兼容
    assert [msg["role"] for msg in tok.seen] == ["system", "user"]


def test_row_reward_fn_switches_by_family():
    m = importlib.import_module("32_run_grpo")
    bar_row = {"family": "barge_in_context", "interrupted_content": "我能说"}
    gate_row = {"family": "low_risk_clear", "response": "好的"}
    assert m._row_reward_fn(bar_row, W) is not None
    assert m._row_reward_fn(gate_row, W) is None
    # bargein 分支确实走 context_recall 口径（复述 vs 无关 分数差显著）
    fn = m._row_reward_fn(bar_row, W)
    hi = fn("我能说", "我能说", "execute", W)
    lo = fn("完全无关的内容", "我能说", "execute", W)
    assert hi > lo + 0.2
    # gate 分支走原 grpo_generation_reward（回归：签名不变仍可调用）
    assert grpo_generation_reward("好的，已执行。", "好的，已执行。", "execute", W) > 0
