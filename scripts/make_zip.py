"""重建 AutoDL 部署包：代码+配置+数据（含真实语料与 TTS 语音输入），排除权重/缓存/大产物。"""
import zipfile
from pathlib import Path

ROOT = Path(r"c:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\trusted-full-duplex-agent")
OUT = Path(r"c:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\trusted-full-duplex-agent-autodl.zip")

EXCLUDE_DIRS = {"weights", "__pycache__", ".pytest_cache", ".git", "outputs"}
EXCLUDE_FILES = {"dl_local_poc_model.py", "tts_experiment.py",
                 "diag_grpo_gen.py", "diag_safety_gen.py"}
EXCLUDE_PREFIXES = tuple(f"scripts\\ssh_" for _ in [1])  # 本地 SSH 工具不上远端

files = []
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        continue
    if rel.name in EXCLUDE_FILES:
        continue
    if str(rel).startswith("scripts\\ssh_"):
        continue
    files.append(rel)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in files:
        z.write(ROOT / rel, Path("trusted-full-duplex-agent") / rel)

print(f"打包 {len(files)} 个文件 -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
key = ["data\\rl_streams\\rl_streams.jsonl", "data\\user_turns\\ambiguous.wav",
       "requirements_minicpmo.txt", "scripts\\31_prep_rl_data.py"]
for k in key:
    print(("  ok " if k in {str(f) for f in files} else "  MISSING ") + k)