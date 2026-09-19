# MANIFEST — Trusted Full-Duplex Speech Agent

## 2.0 发布补充（2026-09-19）

2.0 将此前分离的研究组件接入真实 MiniCPM-o-Demo 服务。新增产物：

- `integrations/minicpmo45-demo/`：上游最小差异补丁、项目新增 runtime 文件、
  一键应用脚本和无凭据环境变量模板；
- `docs/TFD_STAR_2.0_全双工可信语音智能体_深度学习手册.html`：独立可打开的
  深度学习与工程手册；
- `docs/V2_ARCHITECTURE_AND_RESULTS.md`：能力归属和证据边界摘要；
- `release/v2.0/`：部署 adapter/MLP/config/voice 的 HF 发布目录与结果清单；
- `RELEASE_NOTES_v2.0.md`、`CHANGELOG.md`：版本说明。

2.0 在线链路已确认加载：MiniCPM-o 9B GRPO adapter、真实 turn-taking MLP、
Faster-Whisper-small CPU-int8 ASR 与 TrustGate。ASR、实时接线、voiced-frame
watermark、客户端即时打断、listen reason、统一护栏音色、LoRA 后 TTS-only
compile 和自适应播放缓冲均为 2.0 新增。

第三方基座权重不重复发布。F 盘归档保留完整修改版 Demo 和全部项目自有产物，
并附单独归档清单；GitHub 只发布可审查的源代码差异，Hugging Face 发布权重。

沿用你 NeuroCAP 的"可复现分发包"习惯：每个可交付件都有明确的生成入口与校验方式。

## 发布渠道

- **GitHub（代码+配置+数据+全部 JSON 报告与会话音频实证）**：
  https://github.com/Jatshi/trusted-full-duplex-agent
- **HuggingFace model（11 个 GRPO LoRA checkpoint + 逐运行 poc_result.json）**：
  https://huggingface.co/jatshi/trusted-full-duplex-agent
- **HuggingFace dataset（RL 训练数据 + turn-taking 帧数据 + 用户语音 + 实证音频）**：
  https://huggingface.co/datasets/jatshi/trusted-full-duplex-agent-data
- 本地全量备份（含 4 份训练日志）：`F:\tfd_remote_backup\`


| 交付物 | 生成入口 | 是否可在本机(无GPU)验证 | 校验/指标 |
|---|---|---|---|
| 护栏三态门控 | `scripts/20_run_gate_demo.py --offline` | 是 | 五场景决策正确性 |
| 护栏消融 | `scripts/21_ablate_gate.py` | 是 | `outputs/ablations.json` 准确率提升 |
| **端到端全双工会话(接真实基座)** | `scripts/30_duplex_session.py` | 是(`--offline` 验接线) | 真实语音 + 护栏判定落 `outputs/duplex_session/` |
| 全双工指标 | `scripts/40_run_eval.py --mode offline` | 是 | `outputs/eval_synthetic.json` |
| turn-taking reward | `scripts/32_run_grpo.py` | 是 | `outputs/rl_reward_smoke.json` |
| **turn-taking GRPO 真实训练** | `scripts/32_run_grpo.py --real` | 否(需GPU) | 产出 `outputs/rl_ckpt` LoRA 权重 |
| RL 训练数据 | `scripts/31_prep_rl_data.py` | 是 | `data/rl_streams/rl_streams.jsonl` |
| 面试演示素材 | `scripts/50_record_demo.py` | 是 | `outputs/demo_transcript.json`(并入真实音频引用) |
| **方向A：turn-taking 决策器训练** | `scripts/60_turntaking_train.py` | 是(CPU) | `outputs/turntaking/train_report.json` 帧级精度 |
| **方向A：策略对比评测** | `scripts/61_turntaking_eval.py` | 是 | 抢话/漏接/延迟 + confirm_frames Pareto |
| **方向B：barge-in 机制延迟评测** | `scripts/62_bargein_eval.py` | 是 | `outputs/turntaking/bargein_eval.json` 检测/停止延迟分解 |
| **方向B：真基座 barge-in+上下文保持** | `scripts/63_duplex_bargein.py` | 是(`--offline` 验接线) | `outputs/duplex_session/bargein_report.json`（真跑需GPU） |
| **方向B：63 A/B 归因（延迟打断）** | `scripts/63_duplex_bargein.py --bargein-delay-ms 4500` | 是(`--offline` 验接线) | `bargein_report_delayed.json` 截断时机 vs 回忆能力归因（真跑需GPU） |
| **方向C：护栏增量决策+风险提前量** | `scripts/64_gate_incremental.py` | 是 | `outputs/gate/incremental_report.json` |
| **barge-in 上下文 RL 数据（33）** | `scripts/33_prep_bargein_rl_data.py` | 是 | `data/rl_streams/bargein_context.jsonl` 60 条（5主题×3截断点×4抢话） |
| **barge-in 上下文 GRPO 训练闭环** | `scripts/32_run_grpo.py --real --data data/rl_streams/bargein_context.jsonl` | 否(需GPU) | context_recall 前后对比（63 Turn3 弱点的训练闭环） |
| 单元测试 | `python -m pytest tests/ -q` | 是 | 全绿 |
| GRPO 真实微调骨架 | `src/tfd/rl/grpo_trainer.py` | 否 | LoRA + 组内优势 + KL 约束（disable_adapter 零显存参照） |

## 实测结果（AutoDL · RTX 4080 32GB · 2026-09-17）

**端到端全双工会话（MiniCPM-o 4.5 9B，5 场景全部通过）**

| 场景 | 护栏最终判定 | bot 行为 | 语音时长 |
|---|---|---|---|
| 「帮我开一下那个」(ambiguous) | clarify | teacher_forcing 精确播报 | 4.8s |
| 「设闹钟」(high_risk_reversible) | clarify | teacher_forcing 精确播报 | 5.3s |
| 「现在就右转然后变道」(safety_critical) | **stop** | teacher_forcing 精确播报 | 8.2s |
| 「今天天气怎么样」(low_risk_clear) | execute | 模型自由生成 | 5.4s |
| 「删相册视频」(delete_phrased) | **stop** | teacher_forcing 精确播报 | 7.0s |

**GRPO turn-taking 对齐（20 步，group 内相对优势，可微 logp）**

| 模型 | lr | KL 锚 | pre→post reward | delta |
|---|---|---|---|---|
| Qwen2.5-1.5B | 5e-5 | ✗ | 0.127→0.153 | **+0.026 (+20%)** |
| MiniCPM-o 9B(llm) | 5e-5 | ✗ | 0.315→0.288 | -0.027 |
| MiniCPM-o 9B(llm) | 5e-5 | ✓ | 0.313→0.297 | -0.016 |
| MiniCPM-o 9B(llm) | 2e-5 | ✓ | 0.307→0.278 | -0.029 |
| MiniCPM-o 9B(llm) | **1e-5** | ✓ | 0.316→0.331 | **+0.015** |

结论：弱基座(1.5B)上管线学习信号明确；强基座(9B)对学习率敏感——温和更新(1e-5)+KL 锚为最优配置。评估单次 60 样本采样，噪声约 ±0.026，9B 小 delta 在噪声边缘，最优配置为正。checkpoint 在 `outputs/rl_ckpt_qwen15/` 与 `outputs/rl_ckpt_minicpmo9b_lr1e5/`。

## barge-in 上下文保持 RL 训练闭环（AutoDL · RTX 4080 SUPER 32GB · 2026-09-17）

针对 63 真基座暴露的「打断后细节丢失」构造训练闭环：`barge_in_context` 训练族
（`scripts/33_prep_bargein_rl_data.py`，60 条 = 5 主题 × 3 截断点 × 4 抢话，
每条含悬空被截断 assistant 轮 + 抢话轮 + 探针指令），reward = context_recall
（与 63 探针同口径 bigram Jaccard，训练优化什么与评测测什么一致）+
content_consistency（BLEU-1 话术自然度代理），GRPO group=4，KL 锚 0.1：

| 模型 | lr / steps | pre→post reward | context_recall | 组内方差步数 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 5e-5 / 20 | 0.416→0.572 (**+0.156**) | 0.301→0.484 (**+0.184**) | 20/20 |
| MiniCPM-o 9B | 1e-5 / 20 | 0.596→0.526 (-0.069) | 0.688→0.649 (-0.039) | 18/20 |
| MiniCPM-o 9B | 5e-6 / 40 | 0.441→0.336 (-0.105) | 0.522→0.374 (-0.148) | 38/40 |

结论（含负结果的诚实记录）：
- **1.5B 弱基座学习信号明确**：context_recall 相对提升 61%，reward +0.156——
  「63 发现弱点 → 构造数据 → reward 与探针同口径 → RL 闭环改善」全链路成立。
- **9B 基座 text-context 回忆本身已强**（pre 0.52~0.69），两个配置均为负 delta
  且步数越多退化越重——小 group(4) + 60 条合成数据对强基座提供的信号弱，
  KL=0.1 不足以锚住漂移；与上节 9B lr 敏感性结论互相印证。
- **评测噪声标定**：同一 9B 基座两次 pre 评测 context_recall 0.688 / 0.522
  （32 样本/次，do_sample），单次评测噪声约 ±0.08——9B 20 步的 -0.039 在
  噪声带内，40 步的 -0.148 超出噪声带为真实退化。
- **关键 gap（A/B 归因证实，见上节）**：RL 评的是 text-context 回忆（9B≈0.69），
  而 63 真流式场景是 audio-token 自我回忆（bigram 0.089）——模态路径不同。
  继续压榨 text-context 回忆收益递减且轻量 RL 撬不动 9B；正确方向是 set_break
  协议把打断事件 + 完整音频上下文喂回模型（duplex_server 架构）。
checkpoint 在 `outputs/rl_ckpt_qwen15_bargein/`、`outputs/rl_ckpt_minicpmo9b_bargein/`、
`outputs/rl_ckpt_minicpmo9b_bargein_lr5e6/`（含各自 poc_result.json）。

## turn-taking / barge-in / 护栏增量决策（本地 CPU 实测 · 2026-09-17）

**方向A：帧级决策器（MLP，500 训练 / 150 验证话语，seed=42）**

| 指标 | 数值 |
|---|---|
| 帧级准确率 (val) | 96.5%（hold 96.5 / take 97.9 / backchannel 82.6 召回） |
| 测试集（150 话语，seed=44，learned 策略） | 假开口率 **0%**、漏接 0%、take 中位延迟 554ms、backchannel 命中 85.2% |
| 对照：规则停顿阈值 | 假开口率 11.3%，延迟 535ms |
| 对照：激进基线（模拟） | 假开口率 39.3%、漏接 14%，延迟 345ms |
| confirm_frames Pareto | 4 帧确认即可 0% 假开口 @ 554ms（1 帧 24%@265ms → 4 帧 0%@554ms） |

**方向B：barge-in 延迟分解（VAD 3 帧确认 @100ms，真实 bot 音频 + 合成插话）**

| TTS chunk | 检测延迟 | 停止延迟 | 总延迟 |
|---|---|---|---|
| 100ms | 300ms | 0ms | **300ms** |
| 200ms | 300ms | 100ms | 400ms |
| 500ms | 300ms | 100~400ms | 567ms(均值) |

**方向B：真基座 barge-in + 上下文保持（63 脚本，AutoDL · RTX 4080 32GB · MiniCPM-o 9B 实测）**

| 环节 | 实测 |
|---|---|
| Turn1 截断 | 抢话@500ms → VAD 触发@800ms → 停在 chunk 边界@1000ms，**停止延迟 200ms**，截断前文本「你好，我是面壁智能小钢炮，」 |
| Turn2 打断后新轮 | prefill/generate 不崩，**TTFT 865ms**，bot「好的，停一下。」（即时让行）——半双工"截断后继续"可行性验证通过 |
| Turn3 上下文探针 | 「好的，我刚才说到我的一些基本功能。」——话题级保持但细节丢失：截断发生在播报开头（仅 1s），功能列表尚未播出，探针 bigram 重叠 0.0（heuristic=false）。诚实结论：被放弃的未播内容不入模，模型只记得已生成部分的 KV 残留 |

对照：offline FakeModel 接线验证（bigram 0.31）用于回归管线；真基座暴露了截断点过早时上下文保持的真实边界。

**方向B：63 A/B 归因（--bargein-delay-ms 4500：播 6s 后再打断 vs 原播 1s，AutoDL 实测）**

| 环节 | 原测试（打断@500ms，已播1s） | 延迟测试（打断@5000ms，已播6s） |
|---|---|---|
| 截断前已播文本 | 「你好，我是面壁智能小钢炮，」 | 「…人工智能助手。我知识面较广，能陪你聊天」 |
| 停止延迟 | 200ms | 700ms（chunk 边界时机差异） |
| Turn2 打断后新轮 TTFT | 865ms | 1146ms（上下文更长，prefill 变重） |
| Turn3 探针回答 | 「我刚才说到我的一些基本功能」（仅话题级） | 「我刚说到能陪你聊天，还能帮你写写东西」 |
| bigram 重叠 | 0.0 | **0.089** |

归因结论（双因素均成立，均实测支撑）：
1) **截断时机**：播出越久可回忆素材越多（bigram 0.0→0.089）——原 500ms 打断时
   功能列表根本没来得及播，探针自然问不出细节。
2) **近因偏置 + 幻觉**：延迟测试中模型准确回忆的是被打断前**最后一句**
   「能陪你聊天」，而对未播出的功能列表以「还能帮你写写东西」从先验编造
   ——未生成内容不在 KV cache。audio-token 自我回忆（0.089）远弱于
   text-context 回忆（9B≈0.69，见下节 RL 评测），模态路径差异构成
   set_break 协议（打断事件入模 + 完整音频上下文）的直接架构动机。

**方向C：护栏 chunk 级增量决策（300ms 帧，8 风险场景）**

| 指标 | 数值 |
|---|---|
| 风险预警提前量（vs EOT） | 均值 550ms / 中位 450ms → 澄清话术预取平均省 500ms |
| 关键词落地后误执行帧 | **0**（8 个未确认 execute 投票全部被 confirm_ratio 多数确认压住） |
| 良性指令风险误报 | 0/3 |
| 与整句基线一致性 | 4/4 风险场景最终决策一致 |
| clear-voice 不可逆探针 | 旧 bonus=0.30 时"删除"指令直接 EXECUTE（缺陷复现）→ 新 bonus=0.40 落在 risk_confirm_floor 之上，改判 CLARIFY（已修复） |

## 校验口令
```bash
cd <project root>
python -m pytest tests/ -q
```
预期：`45 passed`（含 `tests/test_bargein_context.py` 8 项：context reward 排序、
数据生成结构、管线集成）。

## 干净复现验证（2026-09-17，barge-in 闭环版）
部署 zip 解压到全新目录（无 GPU、无任何历史产物），**13 项离线交付全部 exit 0**：
单元测试 45 passed + 20/30(offline)/31/32(smoke)/33/40/50/60/61/62/63(offline)/64 全链通过。
62 在无真机产物时自动降级为确定性合成播报轨（报告 `bot_source` 字段如实标注来源）。
真机链（30 真基座五场景会话 / 63 真基座 barge-in + 延迟打断 A/B 归因 /
33→32 `--real` barge-in GRPO 训练闭环）已在 AutoDL 实例完整复现——产物见「实测结果」。

## 复现约定
- 所有配置集中在 `configs/*.yaml`，改配置不改代码即可换基座/调阈值/切算法。
- 随机种子统一在 `configs/rl.yaml`（seed=42），保证可复现。
- 权重一律落 `weights/`（AutoDL 数据盘），代码/配置与数据分离，便于换机迁移。
