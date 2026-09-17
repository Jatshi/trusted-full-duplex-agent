"""脚本统一引导：把项目根目录与 src 加进 sys.path，加载配置。

用法： ``from bootstrap import root, load_config``
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tfd.utils.config import load_config, resolve  # noqa: E402

# 开机后显式 XYZ=xxx 可覆盖 configs/base.yaml 的 paths
os.environ.setdefault("TFD_WEIGHTS", "weights")
os.environ.setdefault("TFD_DATA", "data")
os.environ.setdefault("TFD_OUT", "outputs")
for d in ("weights", "data", "outputs", "outputs/eval_reports", "outputs/rl_ckpt",
          "data/rl_streams", "data/eval_wavs"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)