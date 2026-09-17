#!/usr/bin/env python3
"""显存测量：复现 Step B 最终配置（MiniCPM-o 9B 文本路径 + LoRA + group_size=4），跑 2 步真实 GRPO 记录峰值显存。"""
import json
import torch
from bootstrap import load_config, ROOT
from transformers import AutoTokenizer
from tfd.rl.grpo_trainer import load_causallm, GRPOTrainer, SYSTEM_PROMPT

rl_cfg = load_config("rl")
rl_cfg.setdefault("rl", {})["group_size"] = 4
t = rl_cfg["training"]
base = "weights/minicpm-o-4_5"
device = "cuda"

print("加载 tokenizer...")
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("加载 MiniCPM-o 9B 文本路径 + LoRA...")
model = load_causallm(base, device, use_lora=True, train_cfg=t)
n_params = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"参数量: total={n_params/1e9:.2f}B  trainable={n_train/1e6:.1f}M")

torch.cuda.reset_peak_memory_stats()
trainer = GRPOTrainer(rl_cfg, model, None, model, tokenizer=tok)
rows = [json.loads(ln) for ln in
        (ROOT / rl_cfg["data"]["stream_dir"] / "rl_streams.jsonl").read_text(encoding="utf-8").splitlines() if ln][:2]
for i, r in enumerate(rows):
    ids = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": r.get("instruction", "")}],
        add_generation_prompt=True, return_tensors="pt").to(device)
    loss = trainer.train_step_on(ids, [r.get("response", "")], r.get("gate_expected", "execute"))
    print(f"step {i}: loss={loss:.4f}  peak_allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GB")

peak = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"[显存] peak_allocated={peak:.2f}GB  peak_reserved={reserved:.2f}GB  GPU_total={total:.1f}GB")
