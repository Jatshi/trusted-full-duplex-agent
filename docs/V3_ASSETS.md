# 3.0 资产配置

GitHub 源码发布不携带模型权重或个人录音。原有公开 GRPO / MLP / 数据材料见
[模型仓库](https://huggingface.co/jatshi/trusted-full-duplex-agent)
与 [数据仓库](https://huggingface.co/datasets/jatshi/trusted-full-duplex-agent-data)。
六份新 SFT、四个动作头与 18 份记忆权重没有随此次源码发布上传，使用者需提供相应训练产物。

| 资产 | 配置入口 | 契约 |
| :-- | :-- | :-- |
| MiniCPM-o 4.5 | 后端 `--model-path` | 上游模型与许可 |
| GRPO adapter | `TFD_GRPO_ADAPTER` | PEFT LoRA 配置与权重 |
| 声学 MLP | `TFD_TURN_MLP_WEIGHTS` | 10 维逐帧特征 |
| Faster-Whisper | `TFD_ASR_MODEL` | CTranslate2 模型目录 |
| TrustGate | `TFD_GATE_CONFIG` | 本项目 gate.yaml |
| 六份 SFT | `TFD_LORA_PROFILE_MANIFEST` | 文件 SHA 与每份 504 个实际张量校验 |
| 四个动作头 | `TFD_SHADOW_HEAD_DIRECTORY` | 与 demo_shadow.py 的固定 checkpoint 格式一致 |
| EasyTurn / Smart Turn | 语义侧车 `--assets` | 官方模型、前端、配置与 tokenizer |
| 18 份记忆 | 面板目录的 assets.json | 编码缓存、训练数据契约、权重与源码 SHA |

## SFT 清单

在外部 JSON 文件中提供六个模式到 checkpoint 目录的映射，然后生成已钉哈希的清单：

```bash
PYTHONPATH=src python integrations/minicpmo45-demo-v3/prepare_adapters.py /path/to/paths.json /path/to/adapter_manifest.json
```

键为 `v3_lora_e127_s42` / `s43` / `s44` 和 `v3_lora_e134_s42` / `s43` / `s44`。
值为包含 adapter_config.json、adapter_model.safetensors 的绝对目录。
脚本不下载、不训练、不覆盖已有输出。不配置时可使用 engineering / reference 模式。

## 语义模型目录

```text
assets/
  easy_turn_src/Easy_Turn/  官方实现与 examples 配置
  easy_turn/checkpoint.pt
  qwen2_5_0_5b/            官方 Qwen tokenizer / 模型
  smart-turn-v3.2-cpu.onnx
```

语义侧车使用 EasyTurn 自带的环境依赖与 CPU 推理。
下载前核对上游模型卡、许可、修订与可用磁盘；不得用其他 checkpoint 冒充固定模型。

## 记忆面板

准备 E136 的 `data.json`、`encode/features.pt` 及 memory / gru / no_clear 三 seed 权重；
准备 E137 的 memory_endpoint / memory_prefix / gru_prefix 三 seed 权重。
每个权重位于 `<arm>_<seed>/best.pt`。然后：

```bash
python integrations/minicpmo45-demo-v3/sidecars/memory/prepare_assets.py --parent /path/to/e136/run --prefix /path/to/e137
```

清单钉死事件特征、源码及 18 份权重，面板仅使用训练表达，不把测试表达混入缓存。
