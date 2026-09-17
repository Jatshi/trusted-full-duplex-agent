# Trusted Full-Duplex Speech Agent（可信全双工语音智能体）

在开源全双工基座（MiniCPM-o 4.5）之上构建的**系统层全双工 + 可信护栏 + turn-taking 对齐**完整工程：
流式三态护栏（execute/clarify/stop）、帧级 turn-taking 决策器、barge-in 打断处理、
chunk 级增量风险决策、以及「发现弱点 → 构造数据 → GRPO 闭环 → 同口径评测」的 RL 训练闭环。

> 一句话价值：让全双工语音不仅"流畅像人"，还会 **"该执行就执行、该澄清就澄清、该停就停"**，
> 并且每个结论都有实测数字与归因实验支撑——包括负结果与噪声标定。

---

## 实测结果一览（真机 GPU 验证）

| 能力 | 实测指标 | 验证方式 |
|---|---|---|
| **Turn-taking 决策器**（外部轻量分类器，200ms 帧） | 帧级准确率 96.5%，**假开口率 0%**，take 中位延迟 554ms | 训练/测试分离 + 规则基线（11.3% 假开口）与激进基线（39.3%）对照 |
| **Barge-in 打断** | VAD 检测 300ms + chunk 边界停止 0~400ms；真基座截断后新轮 TTFT 865ms 不崩 | 机制层 + MiniCPM-o 9B 真基座双层验证 |
| **流式护栏增量决策** | 风险预警提前量均值 **550ms**（澄清话术预取），关键帧误执行 **0** | 300ms chunk 级增量 vs 整句基线一致性 4/4 |
| **RL 训练闭环**（barge-in 上下文保持） | 1.5B：context_recall 0.301→0.484（**+0.184**）；9B 负结果 + 评测噪声标定 ±0.08 | GRPO 真梯度，reward 与探针评测同口径（bigram Jaccard） |
| **A/B 归因实验** | 推迟打断 500ms→5000ms：探针 bigram 0.0→0.089，证实「截断时机 + 近因偏置」双因素 | 主动设计实验回答自身系统的弱点 |

所有数字的生成入口、复现命令与原始 JSON 报告见 **[MANIFEST.md](MANIFEST.md)**。

## 系统架构

```
用户语音流 ──┬─→ [Turn-taking 决策器] 200ms 帧级 hold/take/backchannel
             │        ↓ take
             ├─→ [流式护栏] chunk 级增量风险决策（risk latch + 提前预警）
             │        ↓ execute/clarify/stop
             └─→ [MiniCPM-o 4.5] streaming_prefill / streaming_generate
                      ↓ 流式播报（可中断）
             [EnergyVAD] 检测用户抢话 → 截断生成 → 打断事件入上下文
                      ↓
             [RL 对齐层] GRPO：turn-taking 时机 + barge-in 上下文保持 reward
```

| 模块 | 位置 | 说明 |
|---|---|---|
| 基座抽象 | `src/tfd/base/backends.py` | MiniCPM-o 流式接口（streaming_prefill / streaming_generate / duplex_server） |
| 流式护栏 | `src/tfd/gate/` | 三态门控、风险 latch、chunk 级增量决策 |
| Turn-taking | `src/tfd/turntaking/` | 帧级决策器训练（MLP）+ 学习型/规则/激进三策略对比 + barge-in VAD |
| RL 对齐 | `src/tfd/rl/` | GRPO trainer（可微 logp + 组内优势 + KL 锚）、turn-taking 与 barge-in context reward |
| 评测 | `src/tfd/eval/` | 假开口率/打断延迟/风险提前量/内容一致性等全双工自定义指标 |

## 快速开始

### 本机（无 GPU）—— 13 项离线交付全部可跑

```bash
pip install -r requirements_minicpmo.txt   # 或最小依赖：numpy pyyaml soundfile pytest
python scripts/20_run_gate_demo.py --offline    # 护栏三态 demo
python scripts/30_duplex_session.py --offline   # 端到端接线冒烟
python scripts/33_prep_bargein_rl_data.py       # barge-in RL 数据生成
python scripts/60_turntaking_train.py           # turn-taking 决策器训练（CPU 即可）
python scripts/61_turntaking_eval.py            # 策略对比 + confirm_frames Pareto
python scripts/62_bargein_eval.py               # barge-in 延迟分解
python scripts/63_duplex_bargein.py --offline   # barge-in 接线验证
python scripts/64_gate_incremental.py           # 护栏增量决策评测
python -m pytest tests/ -q                      # 45 项单元测试
```

### GPU 真机（AutoDL / 24GB+ 显存）

```bash
bash scripts/run_autodl.sh setup        # 官方锁定依赖 + 权重下载（HF 镜像）
bash scripts/run_autodl.sh online       # 真基座端到端会话 + 演示录制
# 真基座 barge-in 与 A/B 归因：
python scripts/63_duplex_bargein.py
python scripts/63_duplex_bargein.py --bargein-delay-ms 4500
# barge-in 上下文 GRPO 训练闭环：
python scripts/32_run_grpo.py --real --model <模型路径> \
  --data data/rl_streams/bargein_context.jsonl --lr 1e-5 --group-size 4
```

## 关键实验与诚实边界

**A/B 归因（Turn3 细节丢失根因，均有实测支撑）**
- 截断时机：打断越早，可回忆素材越少（500ms 打断时功能列表尚未播出，bigram 0.0）；
- 近因偏置 + 幻觉：延迟打断后模型准确回忆被打断前**最后一句**，未播出内容从先验编造（bigram 0.089）；
- 模态 gap：text-context 回忆（9B≈0.69）远强于 audio-token 自我回忆（0.089）——
  这是 `duplex_server` set_break 协议（打断事件入模 + 完整音频上下文）的直接架构动机。

**RL 训练闭环（含负结果的诚实记录）**
- 1.5B 弱基座学习信号明确（context_recall 相对提升 61%）——「发现弱点→构造数据→
  同口径 reward→GRPO 闭环改善」全链路成立；
- 9B 强基座 text-context 回忆已近天花板（0.52~0.69），轻量 GRPO 两个配置均为负
  delta 且步数越多退化越重；同一基座两次 pre 评测相差 0.166，标定单次评测噪声
  约 ±0.08——负结果与噪声带分析比正数字更有信息量。

**定位说明**
- 本项目是**系统层全双工**（可打断/可回溯/实时决策在系统层），基座在模型层为半双工
  使用（streaming_prefill 听 → streaming_generate 说）；原生全双工模型（Moshi 类
  simultaneous listening & speaking）与本项目是同一问题的两条路线，本项目给出了
  「半双工基座 + 系统层全双工」在可控性/可评测性/可插护栏上的实测依据。
- 基座用开源（MiniCPM-o 4.5），自研贡献在护栏/turn-taking/barge-in/评测/RL 对齐层，
  如实标注。

## RL Checkpoints

barge-in 上下文 GRPO 的 LoRA 权重（Qwen2.5-1.5B 与 MiniCPM-o 9B 各配置）与
poc_result.json 见 HuggingFace：

- Model：https://huggingface.co/jatshi/trusted-full-duplex-agent
- Dataset（训练数据+用户语音+实证音频）：https://huggingface.co/datasets/jatshi/trusted-full-duplex-agent-data

或按上方命令复现训练。

## 目录结构

```
configs/        base/gate/rl/eval/turntaking 五份 YAML（改配置不改代码切基座/调阈值）
src/tfd/        base 基座 · gate 护栏 · turntaking 决策 · rl 强化 · eval 指标
scripts/        bootstrap + 00~64 流程脚本（含 AutoDL 部署 run_autodl.sh）
tests/          45 项 pytest（无 GPU 可跑）
data/           RL 样本 jsonl、user_turns 用户语音 wav、turntaking 训练数据
weights/        模型权重（不入库）
outputs/        评测报告/会话音频/RL checkpoint（JSON 报告入库，权重走 HF）
```

## 复现约定

- 所有配置集中在 `configs/*.yaml`；随机种子统一 seed=42（`configs/rl.yaml`）。
- 训练/评测口径一致：barge-in context_recall 用 `tfd.eval.metrics.bigram_overlap`
  （字符 bigram Jaccard），训练 reward 与 63 号探针评测共用同一实现。
- 完整交付物清单与逐项复现入口见 **[MANIFEST.md](MANIFEST.md)**。
