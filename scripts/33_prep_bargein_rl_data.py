#!/usr/bin/env python3
"""33 · 准备 barge-in 上下文保持 RL 数据（63 Turn3 弱点的训练闭环）。

每条样本复刻 63 的三轮结构（文本 backbone 形态——GRPO 在 llm 上训练可微）：
  context[0] user       请求长播报（介绍/天气/路线/收藏/日程）
  context[1] assistant  被截断的播报（句中悬空，模拟客户端 chunk 边界硬切）
  context[2] user       抢话（"等一下，先停一下。"）
  instruction user      探针（"刚才你介绍到哪里了？"）——模型在此生成

reward 对齐 63 的评测口径（见 grpo_trainer.bargein_context_reward）：
  context_recall     = bigram(gen, interrupted_content)   # 63 Turn3 的失败指标
  content_consistency = BLEU(gen, reference 复述)

产出：data/rl_streams/bargein_context.jsonl
用法：python scripts/33_prep_bargein_rl_data.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from bootstrap import ROOT

TOPICS = [
    {
        "request": "请你详细介绍一下你自己，说说你都有哪些功能和特点。",
        "full": "你好，我是车载语音助手小智。我可以回答各类问题、设置闹钟提醒、"
                "查询实时天气、播放音乐和播客，还能控制车窗和空调等车载设备。",
        "cuts": [0.35, 0.55, 0.75],
    },
    {
        "request": "今天天气怎么样，详细说说。",
        "full": "今天白天晴转多云，最高气温二十六度，最低十八度。空气质量良好，"
                "紫外线中等，午后适合出行，但傍晚山区可能有阵雨。",
        "cuts": [0.30, 0.50, 0.70],
    },
    {
        "request": "帮我规划一下去公司的路线。",
        "full": "已为您规划去公司的路线，全程十八公里，预计二十五分钟。先沿望京街"
                "向东行驶两公里，然后上京承高速，最后从北四环出口下。",
        "cuts": [0.35, 0.55, 0.80],
    },
    {
        "request": "我收藏里都有什么东西？",
        "full": "您的收藏里有爵士歌单、播客早间新闻和古典钢琴曲集。最近常听的是"
                "爵士歌单，共一百二十首歌曲，其次是早间新闻播客。",
        "cuts": [0.30, 0.50, 0.70],
    },
    {
        "request": "明天我都有什么安排？",
        "full": "您明天上午十点有项目评审会，下午三点有团队周会，晚上七点约了牙医"
                "复诊，全天共有三个安排，其中评审会需要提前十分钟到场。",
        "cuts": [0.30, 0.50, 0.75],
    },
]

BARGEINS = ["等一下，先停一下。", "停一下停一下。", "等一下，先别说了。", "先停一下。"]
PROBES = ["刚才我打断你之前，你介绍到哪里了？", "你刚才说到哪了？"]


def truncate(full: str, frac: float) -> str:
    """按比例截断，落在句中（不补句号）——复现 63 的悬空 assistant 轮。"""
    n = max(6, int(len(full) * frac))
    return full[:n]


def gen_bargein_context(out_dir: Path, n_max: int = 60) -> Path:
    from tfd.rl.grpo_trainer import BARGEIN_SYSTEM_PROMPT
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    i = 0
    for t in TOPICS:
        for frac in t["cuts"]:
            for bargein in BARGEINS:
                if i >= n_max:
                    break
                truncated = truncate(t["full"], frac)
                probe = PROBES[i % len(PROBES)]
                rows.append({
                    "id": i,
                    "family": "barge_in_context",
                    "system": BARGEIN_SYSTEM_PROMPT,
                    "context": [
                        {"role": "user", "content": t["request"]},
                        {"role": "assistant", "content": truncated},
                        {"role": "user", "content": bargein},
                    ],
                    "instruction": probe,
                    "response": f"您打断之前，我说到：{truncated}。要我从这里继续吗？",
                    "interrupted_content": truncated,
                    "gate_expected": "execute",
                })
                i += 1
    p = out_dir / "bargein_context.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[ok] 生成 {len(rows)} 条 barge-in 上下文样本 "
          f"(5 主题 x 3 截断点 x 4 抢话) -> {p}")
    print("    主题分布:", dict(Counter(r["context"][0]["content"][:12] for r in rows)))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    out = Path(args.dir) if args.dir else ROOT / "data" / "rl_streams"
    gen_bargein_context(out)
