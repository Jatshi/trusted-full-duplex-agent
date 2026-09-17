#!/usr/bin/env bash
# ============================================================
# AutoDL 一键开机流程：数据盘放权重/镜像缓存，尽量少烧实例费。
# 用法：
#   bash scripts/run_autodl.sh setup      # 装官方依赖 + 权重下载(一次)
#   bash scripts/run_autodl.sh offline    # 无 GPU 也能看(护栏/指标/接线验证)
#   bash scripts/run_autodl.sh online     # 上机真跑基座(端到端全双工, 接 GPU)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-offline}"

# AutoDL 非交互 shell 不带 conda PATH：自动探测 python/pip
if ! command -v python >/dev/null 2>&1 && ! command -v pip >/dev/null 2>&1; then
  for P in /root/miniconda3/bin /opt/conda/bin /usr/local/bin; do
    if [ -x "$P/python" ]; then export PATH="$P:$PATH"; break; fi
  done
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
# 国内 AutoDL 直连 huggingface.co 会被墙：默认走镜像（已设环境变量则尊重）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

setup_env() {
  echo "[1/4] 安装 MiniCPM-o 官方运行时依赖(版本已按官方锁定)"
  pip install -r requirements_minicpmo.txt -q

  echo "[2/4] 下载基座权重到数据盘 weights/ (仅首次)"
  python - <<'PY'
import os
os.makedirs("weights/minicpm-o-4_5", exist_ok=True)
marker = "weights/minicpm-o-4_5/config.json"
if os.path.exists(marker):
    print("权重已存在，跳过下载。")
    raise SystemExit(0)
try:
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/MiniCPM-o-4_5",
                      local_dir="weights/minicpm-o-4_5")
    print("OK_WEIGHTS_DOWNLOADED")
except Exception as e:
    print("跳过下载(可离线续跑/手放权重到 weights/minicpm-o-4_5):", e)
PY
  if [ -f weights/minicpm-o-4_5/config.json ]; then
    # 切到本地权重，避免 online 再从 hub 拉一遍
    sed -i 's/load_from_local: false/load_from_local: true/' configs/base.yaml
    echo "已切换 configs/base.yaml -> load_from_local: true"
  fi

  echo "[3/4] 准备 RL 训练数据(已有数据不覆盖)"
  if [ ! -s data/rl_streams/rl_streams.jsonl ]; then
    python scripts/31_prep_rl_data.py -n 300
  else
    echo "data/rl_streams/rl_streams.jsonl 已存在，保留(不覆盖真实数据)。"
  fi

  echo "[4/4] 建输出目录"
  python -c "import pathlib; [pathlib.Path(d).mkdir(parents=True, exist_ok=True) for d in ('outputs/duplex_session','outputs/eval_reports','outputs/rl_ckpt','data/user_turns')]"
  echo "依赖就绪。"
}

run_offline() {
  echo "=== 本机/离线可跑：护栏 demo + 指标 + 接线验证 + 测试 ==="
  python scripts/00_env_check.py
  python scripts/20_run_gate_demo.py --offline
  python scripts/30_duplex_session.py --offline     # 端到端接线冒烟
  python scripts/21_ablate_gate.py
  python scripts/40_run_eval.py --mode offline
  python scripts/32_run_grpo.py
  python scripts/50_record_demo.py
  python -m pytest tests/ -q || true
}

run_online() {
  echo "=== AutoDL GPU：真实基座 + 端到端全双工 + 演示录制 ==="
  echo "注意: 需要真实人声输入请在 data/user_turns/<name>.wav 放 16k 录音;"
  echo "      否则用测试信号占位, 模型仍会真实生成。"
  python scripts/30_duplex_session.py                # 真正拉基座, 产出真实语音+护栏判定
  python scripts/50_record_demo.py                   # 汇总成面试三栏素材(含真实音频引用)
  echo "全双工在线流程完成。素材: outputs/duplex_session/"
  echo "下一步: python scripts/32_run_grpo.py --real 做 turn-taking 微调(需>=24GB)"
}

case "$MODE" in
  setup)   setup_env ;;
  offline) run_offline ;;
  online)  setup_env && run_online ;;
  *) echo "用法: run_autodl.sh [setup|offline|online]" >&2; exit 1 ;;
esac