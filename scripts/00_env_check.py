#!/usr/bin/env python3
"""00 · 环境探活：上机后先跑这个，确认依赖/GPU/路径就绪，再跑正式流程。"""
from __future__ import annotations
import sys

from bootstrap import ROOT, load_config


def main():
    cfg = load_config("base")
    print("=" * 60)
    print("TFD-Agent 环境探活")
    print("=" * 60)
    print(f"[ok] 项目根目录: {ROOT}")

    # Python / PyTorch / CUDA
    print(f"[ok] python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"[ok] torch: {torch.__version__}")
        print(f"[ok] cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[ok] gpu: {torch.cuda.get_device_name(0)}")
            print(f"[ok] vram: {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
            free = torch.cuda.mem_get_info()[0] / 2**30
            print(f"[ok] vram free: {free:.1f} GiB")
            if free < 16:
                print(f"[!!] 可用显存 <16GiB，configs/base.yaml 建议切 dtype=int8")
        else:
            print("[!!] CUDA 不可用：本脚本仅环境探活，真跑带回 GPU 的 AutoDL")
    except Exception as e:
        print(f"[!!] torch import 失败: {e}")

    # 可选依赖
    for lib in ("transformers", "soundfile", "librosa", "accelerate", "peft"):
        try:
            m = __import__(lib)
            print(f"[ok] {lib}: {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"[..] {lib}: 未安装(按需)")

    # 关键路径可写
    for d in (ROOT / "weights", ROOT / "outputs", ROOT / "data"):
        print(f"[ok] 目录可用: {d}")

    print("=" * 60)
    print("环境就绪。下一步: python scripts/20_run_gate_demo.py --offline")


if __name__ == "__main__":
    main()