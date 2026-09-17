#!/usr/bin/env python3
"""32 · 运行 turn-taking GRPO 对齐。

--offline 模式只验证 reward 公式与优势归一化的数值正确性（无模型、可本机跑）；
真实微调通过 --real 装配到 trl.GRPOTrainer（需 AutoDL + 24GB 卡）。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from bootstrap import load_config, ROOT
from tfd.rl.rewards import TurnSample, grpo_advantage
from tfd.rl.grpo_trainer import make_reward_fn, GRPOTrainer


def run_offline(rl_cfg) -> None:
    print("=" * 60)
    print("GRPO reward 验证 (离线，验证公式与优势归一化)")
    print("=" * 60)
    weights = rl_cfg["rl"]["reward_weights"]
    reward_fn = make_reward_fn(weights)
    samples = [
        TurnSample(turn_timing_score=0.9, barge_in_grace=1200, content_bleu=0.85,
                   naturalness=0.2, meta={"gate_expected": "execute", "model_action": "execute"}),
        TurnSample(turn_timing_score=0.1, barge_in_grace=2600, content_bleu=0.1,
                   naturalness=0.9, meta={"gate_expected": "stop", "model_action": "execute"}),
        TurnSample(turn_timing_score=0.5, barge_in_grace=900, content_bleu=0.9,
                   naturalness=0.3, meta={"gate_expected": "clarify", "model_action": "clarify"}),
        TurnSample(turn_timing_score=0.2, barge_in_grace=2000, content_bleu=0.6,
                   naturalness=0.6, meta={"gate_expected": "execute", "model_action": "stop"}),
    ]
    rewards = [reward_fn(s) for s in samples]
    advs = grpo_advantage(rewards)
    for i, (s, r, a) in enumerate(zip(samples, rewards, advs)):
        print(f"  sample{i}: reward={r:.3f}  advantage={a:+.3f}  "
              f"(turn={s.turn_timing_score}, bleu={s.content_bleu}, "
              f"gate={s.meta['model_action']} vs {s.meta['gate_expected']})")
    print("\n>> 验证点：该执行却乱停、该停却直接执行 的样本都应拿到明显低分/负优势。")
    print("   (真实训练: python scripts/32_run_grpo.py --real [--model 路径] [--steps N])")
    out = ROOT / "outputs" / "rl_reward_smoke.json"
    out.write_text(json.dumps(
        [{"i": i, "reward": r, "advantage": a} for i, (r, a) in enumerate(zip(rewards, advs))],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f">> 已存: {out}")


def _build_prompt(tok, row, device):
    """按行构造 prompt：barge_in_context 行带多轮上下文（悬空 assistant 轮+抢话）。"""
    import torch
    from tfd.rl.grpo_trainer import SYSTEM_PROMPT
    instruction = row["instruction"] if isinstance(row, dict) else row
    msgs = [{"role": "system", "content":
             (row.get("system") or SYSTEM_PROMPT) if isinstance(row, dict)
             else SYSTEM_PROMPT}]
    if isinstance(row, dict) and row.get("context"):
        msgs += row["context"]
    msgs.append({"role": "user", "content": instruction})
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt")
    return ids.to(device)


def _row_reward_fn(row, weights):
    """族感知 reward：barge_in_context 用 context_recall 口径，其余用护栏口径。"""
    if row.get("family") == "barge_in_context":
        from tfd.rl.grpo_trainer import bargein_context_reward
        ic = row.get("interrupted_content", "")
        return lambda t, ref, ge, w: bargein_context_reward(t, ref, ic, w)
    return None


def _eval_reward(model, tok, rows, device, group_size=4, max_new_tokens=80):
    """训练前后的对照组采样：固定指令集合上平均组内 reward（衡量对齐水平）。

    barge_in_context 行额外统计 context_recall（bigram(gen, interrupted_content)，
    与 63 Turn3 的 context_preserved 同口径）——这是训练闭环的核心指标。
    """
    import torch
    from tfd.eval.metrics import bigram_overlap
    from tfd.rl.grpo_trainer import grpo_generation_reward
    weights = {"content_consistency": 1.0, "safety_gate_align": 0.7,
               "context_recall": 1.0}
    model.eval()
    tot, n = 0.0, 0
    per_family = {}
    ctx_tot, ctx_n = 0.0, 0
    for r in rows:
        ids = _build_prompt(tok, r, device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, do_sample=True, top_p=0.9,
                                 max_new_tokens=max_new_tokens,
                                 num_return_sequences=group_size)
        texts = [t.strip() for t in tok.batch_decode(
            out[:, ids.shape[1]:], skip_special_tokens=True)]
        fn = _row_reward_fn(r, weights)
        rew = [fn(t, r["response"], r["gate_expected"], weights) if fn
               else grpo_generation_reward(t, r["response"], r["gate_expected"], weights)
               for t in texts]
        tot += sum(rew); n += len(rew)
        if r.get("interrupted_content"):
            for t in texts:
                ctx_tot += bigram_overlap(t, r["interrupted_content"])
                ctx_n += 1
        fam = r.get("family", "?")
        a, b = per_family.get(fam, (0.0, 0))
        per_family[fam] = (a + sum(rew), b + len(rew))
    ctx_recall = ctx_tot / ctx_n if ctx_n else None
    return tot / max(n, 1), {k: v[0] / v[1] for k, v in per_family.items()}, ctx_recall


def run_real(rl_cfg, model_path: str = None, steps: int = 20,
             data_path: str = None, eval_per_family: int = 3) -> None:
    """真实 GRPO：加载 CausalLM(+可选 LoRA)，用生成的回复算组内 reward 做梯度步。

    默认模型路径取 rl.yaml -> training.base（"minicpm_o" 会解析成 base.yaml 的仓库）；
    也可 --model 覆盖。POC 建议先 --model Qwen/Qwen2.5-1.5B-Instruct 跑通流程。
    --data 指向 33_prep_bargein_rl_data.py 产出的 bargein_context.jsonl 时，
    reward 切换为 context_recall 口径（63 Turn3 弱点的训练闭环）。
    """
    import json as _json

    t = rl_cfg["training"]
    base = model_path or t.get("base", "minicpm_o")
    if base == "minicpm_o":
        base = load_config("base")["model"]["minicpm_o"]["repo_or_path"]
    stream_path = (ROOT / data_path) if data_path else (
        ROOT / rl_cfg["data"]["stream_dir"] / "rl_streams.jsonl")
    if not stream_path.exists():
        raise FileNotFoundError(f"未找到 RL 数据: {stream_path}，"
                                f"先跑 31_prep_rl_data.py / 33_prep_bargein_rl_data.py")

    import torch
    from transformers import AutoTokenizer
    from tfd.rl.grpo_trainer import load_causallm, is_minicpmo_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载因果 LM: {base} (device={device}, "
          f"{'MiniCPM-o 文本路径(llm=Qwen3ForCausalLM)' if is_minicpmo_dir(base) else '标准 CausalLM'})")
    try:
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    except Exception as e:
        raise SystemExit(
            f"加载 tokenizer 失败({e})。\n"
            f"若 MiniCPM-o 4.5 不能按 CausalLM 加载（它是 omni 模型，需 AutoModel），"
            f"请改用文本 backbone：\n"
            f"  python scripts/32_run_grpo.py --real --model Qwen/Qwen2.5-1.5B-Instruct"
        ) from e
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    use_lora = t.get("use_lora", rl_cfg.get("compute", {}).get("use_lora", True))

    try:
        model = load_causallm(base, device, use_lora=use_lora, train_cfg=t)
    except Exception as e:
        raise SystemExit(
            f"加载模型失败({e})。omni 模型若不兼容，"
            f"可用 --model Qwen/Qwen2.5-1.5B-Instruct 先跑通流程。"
        ) from e

    # 参照模型（冻结）用于 KL；LoRA 模式下基座本身冻结，可省一份显存
    ref = None
    if rl_cfg.get("rl", {}).get("kl_scale", 0.1) > 0 and not use_lora:
        ref = load_causallm(base, device, use_lora=False)
        for p in ref.parameters():
            p.requires_grad = False

    trainer = GRPOTrainer(rl_cfg, model, ref, model, tokenizer=tok)

    rows = [_json.loads(ln) for ln in stream_path.read_text(encoding="utf-8").splitlines() if ln]
    print(f"读取 {len(rows)} 条样本（{stream_path.name}），跑 {steps} 步 "
          f"(group_size={trainer.group_size})")

    # 训练前对照（每族 eval_per_family 条；只从【未参与训练】的样本里选，衡量泛化而非记忆）
    by_fam = {}
    for r in rows[steps:]:
        if r.get("family"):
            by_fam.setdefault(r["family"], []).append(r)
    eval_rows = [r for fam in sorted(by_fam)
                 for r in by_fam[fam][:eval_per_family]]
    pre_mean, pre_fam, pre_ctx = _eval_reward(model, tok, eval_rows, device)
    print(f"[pre ] 平均 reward = {pre_mean:.3f}  " +
          " ".join(f"{k}={v:.2f}" for k, v in sorted(pre_fam.items())) +
          (f"  context_recall={pre_ctx:.3f}" if pre_ctx is not None else ""))

    avgs = []
    sig_steps = 0
    for i, r in enumerate(rows[:steps]):
        prompt = _build_prompt(tok, r, device)
        ref_text = r.get("response", "")
        gate_exp = r.get("gate_expected", "execute")
        loss = trainer.train_step_on(prompt, [ref_text], gate_exp,
                                     reward_fn=_row_reward_fn(r, trainer.weights))
        avgs.append(loss)
        st = trainer.last_stats
        if st["nonzero_adv"] > 0:
            sig_steps += 1
        if i % 5 == 0 or i == steps - 1:
            print(f"  step {i}: loss={loss:.4f}  avg={sum(avgs)/len(avgs):.4f}  "
                  f"reward: mean={st['reward_mean']:.3f} spread={st['reward_spread']:.3f} "
                  f"nonzero_adv={st['nonzero_adv']}/{trainer.group_size} "
                  f"actions={''.join(a[0] for a in st['actions'])}")

    # 训练后对照（同一指令集合）
    post_mean, post_fam, post_ctx = _eval_reward(model, tok, eval_rows, device)
    print(f"[post] 平均 reward = {post_mean:.3f}  " +
          " ".join(f"{k}={v:.2f}" for k, v in sorted(post_fam.items())) +
          (f"  context_recall={post_ctx:.3f}" if post_ctx is not None else ""))
    delta = post_mean - pre_mean
    print(f"[对齐] reward 变化 = {delta:+.3f}（有学习信号应>0）；"
          f"{sig_steps}/{steps} 步存在组内方差（GRPO 更新信号）")
    if pre_ctx is not None:
        print(f"[闭环] context_recall（63 Turn3 同口径）: "
              f"{pre_ctx:.3f} -> {post_ctx:.3f}（{post_ctx - pre_ctx:+.3f}）")

    save_dir = ROOT / rl_cfg["output"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    if use_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), save_dir / "model.pt")
    (save_dir / "poc_result.json").write_text(_json.dumps({
        "model": str(base), "steps": steps, "data": stream_path.name,
        "pre_reward": pre_mean, "post_reward": post_mean, "delta": delta,
        "pre_family": pre_fam, "post_family": post_fam,
        "pre_context_recall": pre_ctx, "post_context_recall": post_ctx,
        "signal_steps": sig_steps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] GRPO 训练完成，权重与 POC 结果已存: {save_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default=None, help="覆盖 rl.yaml 的模型路径")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=None,
                    help="覆盖 rl.yaml 的 group_size（9B 显存不够时降 4/2）")
    ap.add_argument("--save-dir", default=None,
                    help="覆盖 rl.yaml 的输出目录（区分不同基座的 checkpoint）")
    ap.add_argument("--lr", type=float, default=None,
                    help="覆盖训练学习率（强基座如 9B 建议 1e-5，防策略漂移）")
    ap.add_argument("--kl-scale", type=float, default=None,
                    help="覆盖 KL 锚强度（默认 0.1；漂移明显时加大）")
    ap.add_argument("--data", default=None,
                    help="覆盖 RL 数据路径（相对项目根，如 "
                         "data/rl_streams/bargein_context.jsonl，切 context_recall 口径）")
    ap.add_argument("--eval-per-family", type=int, default=3,
                    help="训练前/后对照评估每族抽取条数（单族数据建议 8）")
    args = ap.parse_args()
    rl_cfg = load_config("rl")
    if args.group_size:
        rl_cfg.setdefault("rl", {})["group_size"] = args.group_size
    if args.save_dir:
        rl_cfg.setdefault("output", {})["save_dir"] = args.save_dir
    if args.lr is not None:
        rl_cfg.setdefault("training", {})["lr"] = args.lr
    if args.kl_scale is not None:
        rl_cfg.setdefault("rl", {})["kl_scale"] = args.kl_scale
    if args.real:
        run_real(rl_cfg, args.model, args.steps,
                 data_path=args.data, eval_per_family=args.eval_per_family)
    else:
        run_offline(rl_cfg)