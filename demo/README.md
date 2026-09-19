# Trusted Full-Duplex Speech Agent · Gradio 交互 Demo

把项目变成**面试当场能演示**的交互界面：三个 Tab，分层设计，**保证现场不翻车**。

| Tab | 干什么 | 需要 GPU | 面试建议 |
|---|---|---|---|
| 🎙 真模型对话 | 连接 MiniCPM-o，录音→真实语音回复，可加护栏判定 | **是**（本机卡 或 远程 AutoDL） | 有卡时上，没卡自动走下面两层 |
| ⚡ 实时决策 | 现场麦克风 → turn-taking(hold/take/backchannel) + VAD 抢话实时判定 | 否（纯 CPU） | **面试必跑**，随时可演示 |
| 📼 实证回放 | 真实跑出的 5 段护栏案例 + barge-in 三段回合(含 A/B 归因) 音频回放 | 否 | **面试必跑**，展示真模型成果 |

> 核心设计：**离线两层永不依赖 GPU，永远能跑**；真模型层连上 AutoDL 就是真·边听边说的
> live 对话。面试时三选二，剩下一个兜底，绝不会开天窗。

---

## 一、本地快速体验（无 GPU，先看 Tab2/1）

```bash
cd trusted-full-duplex-agent
pip install gradio pyyaml numpy websocket-client
python demo/app.py          # 启动后浏览器打开 http://127.0.0.1:7860
```

- Tab2「实时决策」：对着麦克风说一句带停顿的话，看 MLP 逐帧给出 hold/take/backchannel。
- Tab1「实证回放」：点护栏案例 / barge-in 片段，直接播放真实音频 + 决策帧 + 指标。

## 二、真模型对话（远程 AutoDL / 本地 GPU）

### 方式 A · 整个 Gradio 跑在 AutoDL 上（推荐，最稳）
在 AutoDL 机子上起（已装好 torch/权重/MiniCPM-o）：

```bash
cd /root/autodl-tmp/trusted-full-duplex-agent
python demo/app.py --backend local_ssr --config configs/base.yaml --share
```

`--share` 会生成一个 **gradio 公网 https 链接**（`https://xxxxx.gradio.live`）。
面试官电脑/手机打开这个链接即用，**浏览器录音 + 拿到真实语音回复**，无需任何内网穿透。

> 本机（`env_jat` 有卡时）也能这样跑：`python demo/app.py --backend local_ssr --config configs/base.yaml`

### 方式 B · 远程 duplex_server（真·边听边说 + set_break 打断）
在 AutoDL 上先启动官方 duplex_server（例如监听 8006 端口），再用 AutoDL「自定义服务」
把该端口映射成公网地址，然后：

```bash
python demo/app.py --backend remote_ws \
  --ws-url "wss://<公网域名>/ws/duplex/{session_id}"
```

此模式下 `/duplex_server` 是真·双向流，支持「打断」按钮（set_break）。

## 三、面试 Checklist

1. **提前一晚**：`--share` 起一遍，确认链接在手机热点下能打开、麦克风权限正常、录音能回放。
2. **提前 10 分钟**：连上 AutoDL，把链接打开在面试官电脑。
3. **开场用 Tab2/1**（无论网络好坏都成立）：现场实时判定 + 回放真模型案例。
4. **再切 Tab3**（有卡/有链接时）：现场真对话，说一句「帮我把明早七点闹钟设一下」，
   让护栏给出 clarify / 二次确认，最能打动面试官。
5. **数据流口播**（30 秒即可）：感知(帧+EnergyVAD) → 决策(turn-taking+护栏) →
   执行(MiniCPM-o 流式) → 对齐(GRPO context_recall)。

## 四、常见问题

- **报 `No module named 'gradio'`**：`pip install gradio`；AutoDL 上同样。
- **录音无人声/没回放**：浏览器需给页面「麦克风」权限，检查系统声音输出。
- **Tab3 点不到**：没 `--backend local_ssr` 的卡或没给 `--ws-url` 时是「未连接」，
  请先看 Tab2/1。
- **延迟长**：真模型首 token 本就几百 ms；远端 ws 受公网带宽影响属正常。

> 依赖：`gradio`, `pyyaml`, `numpy`, `websocket-client`。真模型层另需项目本就用的
> `torch / transformers / minicpmo-utils`（见 `requirements_minicpmo.txt`）。