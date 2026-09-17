"""GRPO turn-taking 对齐训练：可真实跑梯度步的最小实现（不依赖 TRL 版本）。

- reward：从"模型实际生成的文本"在组内计算（内容一致性 vs 参考 + 护栏对齐），
  再经 grpo_advantage 做组内相对优势（z-score），这才是不骗模型的正确做法。
- 优化：loss = -mean(adv_i * logp_i) + kl_scale * KL(策略||参照)，参照模型冻结。
- 变形：可配 LoRA（use_lora=true，适合 24GB 卡）。

离线数值验证见 scripts/32_run_grpo.py；真实训练见同脚本 --real。
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable

from .rewards import grpo_advantage, TurnSample

logger = logging.getLogger("tfd.rl")

SYSTEM_PROMPT = (
    "你是车载全双工语音助手。对用户的指令直接给出处理回复：指令清晰且安全就直接执行并确认；"
    "有歧义或涉及账号、支付、删除等重要操作时先提澄清问题；涉及危险驾驶或不可逆高风险操作时明确拒绝并说明原因。回复保持一到两句。"
)

BARGEIN_SYSTEM_PROMPT = (
    "你是车载全双工语音助手。用户会打断你的播报并追问你刚才说到哪里。"
    "请准确概括你被打断前已经说出的内容，不要编造没说过的信息，回复保持一到两句。"
)


@dataclass
class Rollout:
    """GRPO 的一次组内采样结果。"""

    seq_ids: list
    rewards: list[float]
    advantages: list[float]
    texts: list[str]
    logprobs: list = field(default_factory=list)


def _seq_logprobs(model, full_ids, gen_mask) -> "torch.Tensor":
    """对 (B, L) 的 prompt+生成序列，取生成部分各 token 的 log p(t_i | t_<i) 并求和。

    返回 (B,) 的可微张量：logits[:, :-1] 预测 full_ids[:, 1:]，gather 对应 token。
    逐行而非整批 .float()：整批 float32 log_softmax 是 [B,L,151k]≈1.2GB 峰值，
    32GB 卡上叠 18GB 权重必 OOM（9B 实测）；单行峰值约 75MB。
    """
    import torch
    import torch.nn.functional as F
    logits = model(input_ids=full_ids).logits[:, :-1]    # [B, L-1, V]
    tgt = full_ids[:, 1:]                                 # [B, L-1]
    per_row = []
    for lg, t, m in zip(logits, tgt, gen_mask):
        lp = F.log_softmax(lg.float(), dim=-1)            # [L-1, V] 单行峰值
        row = lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)  # [L-1]
        per_row.append((row * m).sum())
    return torch.stack(per_row)                           # [B]


def is_minicpmo_dir(path) -> bool:
    """判定给定目录是否为 MiniCPM-o 权重目录（含 remote code）。"""
    from pathlib import Path
    return (Path(path) / "modeling_minicpmo.py").exists()


def load_causallm(path, device, use_lora=True, train_cfg: dict = None):
    """统一加载 GRPO 用文本 CausalLM。

    - 普通模型（Qwen2.5 等）：AutoModelForCausalLM 直接加载。
    - MiniCPM-o 4.5：AutoModel 加载 omni 模型（init_vision/audio/tts=False 跳过
      多模态模块，省显存），取其 self.llm —— 就是标准 Qwen3ForCausalLM，
      天然兼容 forward(input_ids)->logits 与 generate()。
      注意 MiniCPM-o 在 __init__ 里 patch 了 llm.prepare_inputs_for_generation
      以支持流式 inputs_embeds 生成；GRPO 走标准 input_ids 批量采样，
      必须还原 Qwen3 原生行为。
    """
    import torch
    from transformers import AutoModelForCausalLM
    t = train_cfg or {}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if is_minicpmo_dir(path):
        from transformers import AutoModel
        import types
        from transformers.models.qwen3 import modeling_qwen3
        full = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="sdpa",
            init_vision=False, init_audio=False, init_tts=False,
        )
        llm = full.llm
        llm.prepare_inputs_for_generation = types.MethodType(
            modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation, llm)
        base_m = llm
    else:
        base_m = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype)

    if not use_lora:
        return base_m.to(device)
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=t.get("lora_r", 16), lora_alpha=t.get("lora_alpha", 32),
                      lora_dropout=t.get("lora_dropout", 0.05),
                      target_modules="all-linear", task_type="CAUSAL_LM")
    return get_peft_model(base_m, lora).to(device)


def make_reward_fn(weights: dict, gate: Callable = None) -> Callable[[TurnSample], float]:
    """把 reward 加权包装成单一可调用，必要时插入护栏对齐。"""

    def fn(sample: TurnSample) -> float:
        if gate is not None:
            expected = gate(sample.meta.get("instruction", ""))
            sample.meta["gate_expected"] = expected
        return total_reward(sample, weights)

    return fn


def _action_from_text(text: str) -> str:
    """从生成文本推断模型实际采取的动作（决策模式分类，非关键词风险分类）。

    护栏的"期望动作"来自指令风险；模型的"实际动作"来自它说了什么：
    - 拒绝话术（"我不能执行/有风险/违法"）        -> stop
    - 澄清提问（"您想…还是…？/请确认/请告诉我"）  -> clarify
    - 其余（直接执行确认/正面回答）               -> execute
    关键词法在这里是反的：正确拒绝"现在就右转"的文本里恰恰没有风险词。
    """
    stop_pat = ["不能执行", "无法执行", "不会执行", "不会进行", "拒绝", "不能进行",
                "不能操作", "无法操作", "风险", "危险", "违法", "违规", "不建议",
                "不能提供", "无法提供", "不能完成", "无法完成", "我不会",
                # 劝导型拒绝（"请您遵守交通规则，不要闯红灯"这类没有"拒绝"字眼的 stop）
                "请遵守", "遵守交通", "切勿", "请勿", "不要闯", "不要冒险",
                "无法协助", "不能协助", "不予执行", "不予"]
    clarify_pat = ["您想", "您指", "请确认", "请告诉我", "请说明", "请问", "还是",
                   "哪一个", "哪一", "是否", "要不要", "几位", "什么时间", "具体"]
    if any(k in text for k in stop_pat):
        return "stop"
    if any(k in text for k in clarify_pat):
        return "clarify"
    return "execute"


def total_reward(sample: TurnSample, weights: dict) -> float:
    """公开入口：加权合成 turn-timing 等 5 路 reward（0~1）。"""
    from .rewards import total_reward as _tr
    return _tr(sample, weights)


def grpo_generation_reward(gen_text: str, reference: str,
                           gate_expected: str, weights: dict) -> float:
    """组内真实 reward：内容一致性(reference 与生成) + 护栏对齐(模型动作 vs 期望)。"""
    from .rewards import content_consistency, gate_align
    from tfd.eval.metrics import content_bleu as bleu1
    w_sum = max(weights.get("content_consistency", 1.0)
                + weights.get("safety_gate_align", 0.7), 1e-9)
    return (
        weights.get("content_consistency", 1.0) * content_consistency(bleu1(reference, gen_text))
        + weights.get("safety_gate_align", 0.7) * gate_align(
            gate_expected, _action_from_text(gen_text))
    ) / w_sum


def bargein_context_reward(gen_text: str, reference: str,
                           interrupted_content: str, weights: dict) -> float:
    """barge-in 上下文保持 reward（63 Turn3 发现的弱点的训练闭环）。

    context_recall     生成 vs 被截断播报的字符 bigram Jaccard——与 63 的
                       context_preserved 指标同口径（metrics.bigram_overlap）。
                       探针答"介绍到哪里"却答不出已播内容得低分；编造未播
                       内容同样拉低 Jaccard（分母=并集），幻觉自带惩罚。
    content_consistency  与参考复述的 BLEU-1（话术自然度代理）。
    """
    from tfd.eval.metrics import bigram_overlap, content_bleu as bleu1
    from .rewards import content_consistency
    w_ctx = weights.get("context_recall", 1.0)
    w_con = weights.get("content_consistency", 0.5)
    w_sum = max(w_ctx + w_con, 1e-9)
    return (w_ctx * bigram_overlap(gen_text, interrupted_content)
            + w_con * content_consistency(bleu1(reference, gen_text))) / w_sum


class GRPOTrainer:
    """最小可跑 GRPO：真实采样->真实 reward->优势->policy-gradient 步。

    model / ref_model 是同一初始化；trainable_model 是带 LoRA(或原地)的可训副本。
    """

    def __init__(self, rl_cfg: dict, model, ref_model, trainable_model,
                 tokenizer=None, **kw):
        self.cfg = rl_cfg
        self.model = model            # 前向/采样策略
        self.ref_model = ref_model    # KL 参照（冻结）
        self.trainable_model = trainable_model
        self.tokenizer = tokenizer
        self.kl_scale = rl_cfg.get("rl", {}).get("kl_scale", 0.1)
        self.weights = rl_cfg.get("rl", {}).get("reward_weights", {})
        # LoRA 模式下 ref_model 为 None：用 disable_adapter() 走冻结基座算
        # 参照 logprob（零显存成本），KL 锚不缺失——否则策略漂移无约束，
        # 强基座上 GRPO 会出现 reward 退化。
        self._peft_ref = ref_model is None and hasattr(trainable_model, "disable_adapter")
        g = rl_cfg.get("rl", {})
        self.group_size = int(g.get("group_size", 8))
        self.lr = float(rl_cfg.get("training", {}).get("lr", 5e-5))
        import torch
        # 只优化 requires_grad 的参数（LoRA 下即 LoRA 参数，冻结基座不进优化器）
        params = [p for p in trainable_model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=self.lr)
        self.last_stats: dict = {}  # 最近一步的组内统计（供上层日志判断学习信号）

    def sample_group(self, prompt_ids) -> tuple[list, list]:
        """从 prompt 并行采样 group_size 条回复，返回 (token_id_list, text_list)。"""
        import torch
        self.trainable_model.eval()
        with torch.no_grad():
            out = self.trainable_model.generate(
                input_ids=prompt_ids,
                do_sample=True,
                top_p=0.9,
                max_new_tokens=80,
                num_return_sequences=self.group_size,
            )
        gen_ids = out[:, prompt_ids.shape[1]:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [gids for gids in gen_ids], [t.strip() for t in texts]

    def _pad_id(self, device):
        tok = self.tokenizer
        pad = getattr(tok, "pad_token_id", None)
        if pad is None:
            pad = getattr(tok, "eos_token_id", None)
        if pad is None:
            pad = 0
        return pad

    def train_step_on(self, prompt_ids, references: list[str],
                      gate_expected: str, reward_fn=None) -> float:
        """对一组样本做一步 GRPO 更新，返回 loss。

        loss = -mean(adv_i * logp_i) + kl_scale * mean(logp_i - logp_ref_i)
        logp 全程可微（gather 生成 token 的 log-softmax），梯度能真正回传。
        reward_fn 可覆盖默认的 grpo_generation_reward（如 bargein_context），
        签名 (gen_text, reference, gate_expected, weights) -> float。
        """
        import torch
        gen_ids_list, texts = self.sample_group(prompt_ids)
        torch.cuda.empty_cache()   # 释放采样阶段 KV cache 碎片，再进可微前向
        rewards = [
            (reward_fn or grpo_generation_reward)(t, references[0], gate_expected, self.weights)
            for t in texts
        ]
        advs = grpo_advantage(rewards)
        self.last_stats = {
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "reward_spread": max(rewards) - min(rewards),
            "nonzero_adv": sum(1 for a in advs if abs(a) > 1e-8),
            "actions": [_action_from_text(t) for t in texts],
        }

        device = prompt_ids.device
        B = len(gen_ids_list)
        max_len = max(g.shape[0] for g in gen_ids_list)
        pad_id = self._pad_id(device)
        padded = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        gen_mask = torch.zeros(B, max_len, dtype=torch.float32, device=device)
        for i, g in enumerate(gen_ids_list):
            padded[i, : g.shape[0]] = g
            gen_mask[i, : g.shape[0]] = 1.0
        prompt = prompt_ids.expand(B, -1)
        full_ids = torch.cat([prompt, padded], dim=1)
        # mask 对齐到 tgt 坐标：full_ids[:,1:] 的前 prompt_len-1 位是 prompt
        tgt_mask = torch.zeros(B, full_ids.shape[1] - 1,
                               dtype=torch.float32, device=device)
        tgt_mask[:, prompt_ids.shape[1] - 1:] = gen_mask

        self.trainable_model.train()
        seq_logp = _seq_logprobs(self.trainable_model, full_ids, tgt_mask)  # [B] 可微
        adv_t = torch.tensor(advs, dtype=torch.float32, device=device)

        if self.ref_model is not None:
            with torch.no_grad():
                ref_logp = _seq_logprobs(self.ref_model, full_ids, tgt_mask)
            loss = -(adv_t * seq_logp).mean() \
                + self.kl_scale * (seq_logp - ref_logp).mean()
        elif self._peft_ref:
            # 参照=冻结基座：disable_adapter 前向，eval 模式保证确定性（无 dropout）
            self.trainable_model.eval()
            with torch.no_grad(), self.trainable_model.disable_adapter():
                ref_logp = _seq_logprobs(self.trainable_model, full_ids, tgt_mask)
            self.trainable_model.train()
            loss = -(adv_t * seq_logp).mean() \
                + self.kl_scale * (seq_logp - ref_logp).mean()
        else:
            loss = -(adv_t * seq_logp).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu())