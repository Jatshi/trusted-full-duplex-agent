# TFD-STAR 2.0 F 盘归档说明

归档日期：2026-09-19

归档根目录：`F:\博士内容归档\GitHub开源项目补充\TFD-STAR-v2.0`

## 目录内容

- `trusted-full-duplex-agent/`：TFD-STAR 2.0 主仓库完整本地副本，包含源码、测试、实验报告、发布物、学习手册与 Git 历史。
- `MiniCPM-o-Demo-TFD-v2.0/`：经过 TFD-STAR 集成和界面改造的完整 MiniCPM-o Demo 工作副本，用于复现实机全双工演示。
- `remote-runtime-evidence/TFD_STAR_v2_remote_owned_20260919.tar.gz`：从 AutoDL 拉回的项目自有远端代码、运行日志与实验输出归档。
- `remote-runtime-evidence/extracted/`：上述远端归档的可直接浏览版本。

## 远端归档校验

远端归档 SHA-256：

```text
9e7d9298fbc255c289ffa65df8dc0c105c3c54f1ed201f0f2c1c67d26ace7188
```

归档刻意排除了可从官方来源重新下载的第三方大模型权重与缓存，包括 MiniCPM-o 9B、Qwen、Faster-Whisper 基础权重、`node_modules`、Python 缓存。这样既避免重复占用移动硬盘，也避免重新分发第三方模型。项目自有的语轮 MLP、GRPO LoRA adapter、配置、声音资产、日志和评测结果均已保留。

## 公开发布位置

- GitHub 源码与 v2.0.0 Release：`https://github.com/Jatshi/trusted-full-duplex-agent`
- Hugging Face 模型资产：`https://huggingface.co/jatshi/trusted-full-duplex-agent`
- Hugging Face 数据资产：`https://huggingface.co/datasets/jatshi/trusted-full-duplex-agent-data`

## 关键文档

- `trusted-full-duplex-agent/docs/TFD_STAR_2.0_全双工可信语音智能体_深度学习手册.html`
- `trusted-full-duplex-agent/docs/V2_ARCHITECTURE_AND_RESULTS.md`
- `trusted-full-duplex-agent/RELEASE_NOTES_v2.0.md`
- `trusted-full-duplex-agent/integrations/minicpmo45-demo/README.md`

## 验证记录

- 主项目 Python 测试：49 passed。
- Demo Python 集成测试：10 passed。
- 浏览器端 JavaScript 合约测试：10 passed。
- HTML 目录锚点与本地资源检查：通过。
- Demo 集成补丁反向校验：通过。

## 安全说明

SSH 密码、Hugging Face token、GitHub token 等凭据未写入仓库、学习文档、发布包或此归档说明。公开仓库只包含可复现所需的项目资产和配置模板。
