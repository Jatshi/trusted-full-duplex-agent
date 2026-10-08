"""Experimental action-head loader; unset environment preserves 2.0."""
import os
import sys
from pathlib import Path


def build_native_action_bridge_from_env(model):
    checkpoint = os.environ.get("TFD_NATIVE_ACTION_HEAD", "").strip()
    if not checkpoint:
        return None
    source = Path(os.environ.get("TFD_PROJECT_ROOT", "/root/autodl-tmp/trusted-full-duplex-agent")) / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"TFD source directory missing: {source}")
    if str(source.resolve()) not in sys.path:
        sys.path.insert(0, str(source.resolve()))
    from tfd.duplex_policy.native_action_bridge import load_native_action_bridge
    return load_native_action_bridge(model.duplex, checkpoint,
                                     mode=os.environ.get("TFD_NATIVE_ACTION_MODE", "residual"))
