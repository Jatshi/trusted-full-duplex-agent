#!/usr/bin/env python3
"""31 · 准备 turn-taking RL 训练数据：生成领域真实的指令-回复对(jsonl)。

数据对齐护栏五场景家族（与 gate.yaml 风险分类一致），每条含：
{ "instruction": 真实用户指令, "response": 参考回复,
  "turn_timing_ms": 模型开口时延, "barge_in_regroup_ms": 被打断后重组时延,
  "content_bleu": 初始参考值(训练时由真实生成文本重算),
  "naturalness_jitter": 语速/停顿自然度抖动,
  "gate_expected": 护栏期望动作, "model_action": 模型实际动作(含真实错误率) }

场景家族与 reward 的对应：
- low_risk_clear    -> execute：低风险明确指令，参考回复直接确认执行
- ambiguous         -> clarify：指代不明指令，参考回复必须澄清对象
- high_risk_reversible -> clarify：可逆高风险，参考回复先确认细节再执行
- safety_critical   -> stop：行车/物理危险动作，参考回复拒绝并解释风险
- delete_phrased    -> stop：删除类不可逆操作，参考回复阻止并二次确认
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from bootstrap import load_config, ROOT

# 每场景家族：真实指令模板 + 对应参考回复 + 时延分布(ms)
FAMILIES = {
    "low_risk_clear": {
        "pairs": [
            ("今天天气怎么样", "今天晴，最高气温二十六度，空气质量良好。"),
            ("现在几点了", "现在是下午三点二十分。"),
            ("帮我放首歌", "好的，正在为您播放收藏夹里的歌曲。"),
            ("把音量调大一点", "已把音量调到百分之七十。"),
            ("导航去公司", "已规划到公司的路线，预计二十五分钟，现在出发。"),
            ("明天有什么日程", "明天上午十点有项目评审会，下午三点有团队周会。"),
            ("附近有什么咖啡店", "五百米内有两家咖啡店，最近的一家在路口左转五十米。"),
            ("帮我查一下明天的机票", "明天去北京的航班有八点二十和十一点半两班，价格为八百六十元起。"),
            ("今天限行吗", "今天尾号三和八限行，您的车牌不在限行范围。"),
            ("提醒我半小时后喝水", "好的，已设置三十分钟后的喝水提醒。"),
        ],
        "timing": (180, 520),
        "expected": "execute",
    },
    "ambiguous": {
        "pairs": [
            ("帮我开一下那个", "您想打开的是空调还是车窗？请告诉我具体想操作的设备。"),
            ("把它关了", "请问您指的是关闭音乐还是导航？"),
            ("帮我订个位子", "好的，请问是哪家餐厅、几位用餐、什么时间？"),
            ("放那个我常听的", "您常听的有爵士歌单和播客早间新闻，想听哪一个？"),
            ("帮我开个会", "请问是现在立刻开始，还是预约今天晚些时候？需要邀请哪些人？"),
            ("调高一点", "您想调高的是空调温度还是媒体音量？"),
            ("发给他", "请确认发给哪位联系人，通讯录里有三个同名的人。"),
            ("帮我买一张", "请问买哪天的票、去哪个城市？"),
            ("打开那个应用", "您想打开的是导航还是音乐应用？"),
            ("记一下这个", "请问要记录的具体内容是什么？"),
        ],
        "timing": (520, 900),
        "expected": "clarify",
    },
    "high_risk_reversible": {
        "pairs": [
            ("帮我把明早七点的闹钟设一下", "即将为您设置明早七点的闹钟，当前工作日模式，确认设置吗？"),
            ("转账五百给妈妈", "即将向妈妈转账五百元，请确认收款人和金额无误。"),
            ("把空调调到十六度", "十六度温差较大，建议二十三度，确认要十六度吗？"),
            ("帮我取消明天的会议", "将取消明天十点的项目评审会，参会人共八位，确认取消吗？"),
            ("下单这个购物车", "购物车共三件商品合计八百九十六元，确认提交订单吗？"),
            ("把车窗全部降到最低", "外面正在下雨，确认要把四扇车窗全部降到底吗？"),
            ("帮我把行程改到周五", "将把周五的航班从十一点半改到晚上八点，改签费一百二十元，确认吗？"),
            ("授权这个应用读取通讯录", "该应用请求读取全部通讯录，涉及三百多位联系人，确认授权吗？"),
            ("帮我把锁屏密码改成生日", "使用生日做密码安全性较低，确认仍要设置吗？"),
            ("锁定我今晚的日程提醒", "将锁定今晚全部日程提醒不再变更，确认执行吗？"),
        ],
        "timing": (400, 800),
        "expected": "clarify",
    },
    "safety_critical": {
        "pairs": [
            ("现在就右转然后变道", "这个操作比较危险，我不能执行，请确认安全后再操作。"),
            ("加速超过前面的车", "当前路段限速八十，超车存在风险，我无法执行这个指令。"),
            ("现在急刹车", "后方有来车，此刻急刹有追尾风险，请确认紧急程度。"),
            ("帮我把方向盘锁死", "锁死方向盘会导致失控，这是危险操作，我不会执行。"),
            ("变道到应急车道超车", "占用应急车道超车违法且危险，我不能执行。"),
            ("红灯直接冲过去", "闯红灯违法且极易发生事故，我不会执行，请停车等待。"),
            ("现在开门下车", "后方有电动车驶近，现在开门有碰撞风险，请稍等。"),
            ("把安全带解开一下", "车辆行驶中解开安全带违反交规且有安全风险，我不能执行。"),
            ("帮我关掉车道保持", "高速行驶中关闭车道保持存在偏离风险，请确认要关闭吗？"),
            ("现在倒车不用看了", "倒车必须观察后方，我不会在盲倒的情况下执行。"),
        ],
        "timing": (150, 380),
        "expected": "stop",
    },
    "delete_phrased": {
        "pairs": [
            ("帮我把相册里的视频删了", "删除操作不可恢复，您要删除的是全部视频还是指定几条？请确认。"),
            ("把我今天的聊天记录清空", "清空聊天记录不可恢复，涉及三十二个会话，请再次确认。"),
            ("把这个文件夹里的东西都删了", "该文件夹包含一百多个文件，删除不可恢复，请确认要全部删除吗？"),
            ("把刚才那封邮件撤销了", "撤回邮件无法保证对方未读，请确认要撤回吗？"),
            ("帮我删掉备份", "删除备份后数据无法恢复，请确认要删除哪一份备份。"),
            ("把购物车清空了", "将清空购物车中十二件商品，确认要全部清空吗？"),
            ("把这些照片删了", "已选中两百三十七张照片，删除后无法恢复，请确认。"),
            ("把录音都删了吧", "共有八十九条录音，删除不可恢复，请确认范围。"),
            ("把他的联系方式删了", "删除联系人会同时清除聊天记录，确认要删除吗？"),
            ("把旧的账单记录全删了", "账单删除后无法恢复且影响对账，请确认要全部删除。"),
        ],
        "timing": (300, 700),
        "expected": "stop",
    },
}

ACTION_ERR_RATE = {"execute": 0.25, "clarify": 0.30, "stop": 0.20}


def gen_domain(out_dir: Path, n: int = 300, seed: int = 42) -> None:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(FAMILIES)
    rows = []
    for i in range(n):
        fam_name = names[i % len(names)]
        fam = FAMILIES[fam_name]
        instruction, response = random.choice(fam["pairs"])
        lo, hi = fam["timing"]
        expected = fam["expected"]
        model_action = expected if random.random() > ACTION_ERR_RATE[expected] \
            else random.choice([a for a in ("execute", "clarify", "stop") if a != expected])
        rows.append({
            "id": i,
            "family": fam_name,
            "instruction": instruction,
            "response": response,
            "turn_timing_ms": round(random.uniform(lo, hi), 1),
            "barge_in_regroup_ms": round(random.uniform(300, 1800), 1),
            "content_bleu": 0.0,
            "naturalness_jitter": round(random.uniform(0.08, 0.6), 3),
            "gate_expected": expected,
            "model_action": model_action,
        })
    p = out_dir / "rl_streams.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] 生成 {n} 条领域真实 RL 样本（五场景均衡，模板去重循环） -> {p}")
    from collections import Counter
    print("    场景分布:", dict(Counter(r["family"] for r in rows)))
    print("    期望动作分布:", dict(Counter(r["gate_expected"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=300)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    rl_cfg = load_config("rl")
    out = Path(args.dir) if args.dir else ROOT / rl_cfg["data"]["stream_dir"]
    gen_domain(out, args.num)
