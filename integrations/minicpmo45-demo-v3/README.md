# TFD-STAR 3.0 Demo 集成

该目录提供固定上游修订的补丁和 owned overlay。不是上游 Demo 的完整镜像。
使用干净 checkout，安装器拒绝覆盖已有 overlay，并校验 commit、补丁及文件 SHA。

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git /path/to/demo
git -C /path/to/demo checkout 47709a9210dfd71afa76c058e017fc8c4db5c8d2
python apply_integration.py /path/to/demo --check
python apply_integration.py /path/to/demo
```

安装项目 requirements.txt 与上游 Demo requirements。复制 `tfd-runtime.env.example` 到 Git 之外，
填写模型、项目与 Demo 的绝对路径；可选模型按 [资产配置](../../docs/V3_ASSETS.md)准备。

## 服务

在三个终端中启动。示例针对 Linux；端口固定为样例中的 loopback 端口：

```bash
set -a
source /path/to/tfd-runtime.env
set +a
cd "$TFD_DEMO_ROOT"
python -m py_backend.server --host 127.0.0.1 --port 22600 --gpu-id 0 --model-path /path/to/minicpm-o-4_5
```

```bash
cd /path/to/demo
python worker.py --host 127.0.0.1 --port 22640 --gpu-id 0 --backend-server-url http://127.0.0.1:22600
```

```bash
cd /path/to/demo
python gateway.py --host 127.0.0.1 --port 8016 --internal-port 8017 --http
curl -fsS -X PUT http://127.0.0.1:8017/internal/workers/tfdv2-0 -H 'content-type: application/json' --data '{"endpoint":"127.0.0.1:22640","gpu_group":"gpu-0"}'
```

访问 `http://127.0.0.1:8016/audio_duplex`。远端部署使用 SSH 隧道把 8016 转到本机 18006。
后端持有 GPU 时不要启动争抢训练任务；先检查 `/health` 与活动会话，再做单次真实发声预热。
静音健康检查不能替代 TTS 就绪检查。

## 可选 CPU 侧车

```bash
PYTHONPATH=/path/to/project/src python sidecars/observer_service.py --assets /path/to/semantic_assets
cd sidecars/memory
python service.py
```

语义侧车应使用已安装 EasyTurn 官方依赖的环境，服务端口 22680；记忆面板使用 PyTorch 环境，端口 22681。
浏览器的记忆 iframe 默认访问本机 18007；远端部署需额外把 22681 隧道到 18007，
本机运行也需相应转发或修改 iframe 地址。两者默认绑定 loopback，不新增公网服务。

## 模式与恢复

Engineering 是默认交互模式；reference 使用发布 2.0 服务端路径。shadow / components 为实际推理旁路，
六个 LoRA 模式独占主回答适配器。选择模式后重新建立会话，结束后恢复默认 adapter。
未配置实验 checkpoint 时不要选择对应模式；加载失败会明确报错。

环境默认不启用 raw Silero 活跃取消，不启用受控近端回放。
响应跟踪、生命周期和客户端取消是独立配置；请保留恢复环境文件与上游原始 checkout。

## 源码发布内容

补丁、owned Python / JS、组件测试、语义侧车和记忆面板。没有第三方大模型、训练权重、
运行日志、凭据或私人录音。上游 Demo 与第三方资产继续遵循各自许可。
