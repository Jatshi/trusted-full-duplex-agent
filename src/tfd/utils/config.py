"""统一 YAML 配置加载。"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


def _load(file_path: str) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _project_root() -> Path:
    # config.py 位于 <root>/src/tfd/utils/config.py -> parents[3] 为项目根
    return Path(__file__).resolve().parents[3]


def load_config(name: str) -> dict:
    """按名字加载 base/gate/rl/eval/turntaking 配置。"""
    allow = {"base", "gate", "rl", "eval", "turntaking"}
    if name not in allow:
        raise KeyError(f"unknown config group: {name}, allowed={allow}")
    cfg_path = _project_root() / "configs" / f"{name}.yaml"
    return _load(str(cfg_path))


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 进 base。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(path: str) -> str:
    """把相对路径解析为相对项目根目录的绝对路径（AutoDL 上也成立）。"""
    p = Path(path)
    if p.is_absolute():
        return str(path)
    return str(_project_root() / p)