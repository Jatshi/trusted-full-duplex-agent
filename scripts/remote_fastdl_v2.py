#!/usr/bin/env python3
"""pip 依赖安装 v2（简单可靠版）：
本地 torch wheel + tuna 源顺序安装，带重试，日志到 stdout（nohup 重定向 fastdl.log）。
完成标志：FINAL: VERIFY_OK
"""
import glob
import os
import subprocess
import sys
import time

REQ = "/root/autodl-tmp/trusted-full-duplex-agent/requirements_minicpmo.txt"
PIP = "/root/miniconda3/bin/pip"
INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"


def log(*a):
    print(*a, flush=True)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    log("=== 1. 清理一切残留 pip/curl（幂等）===")
    sh("pkill -f pip_retry.sh; pkill -f finish_pip.sh; pkill -f 'pip install'; "
       "pkill -f 'curl -L'; sleep 2")
    sh("ps aux | grep -E 'pip install|finish_pip' | grep -v grep || echo ALL_CLEAN")

    torch_wheels = glob.glob("/root/autodl-tmp/torch-2.8.0*.whl")
    if not torch_wheels:
        log("FATAL: 本地 torch wheel 不见了")
        sys.exit(1)
    wheel = torch_wheels[0]
    log(f"=== 2. 安装本地 torch wheel: {wheel} "
        f"({os.path.getsize(wheel)/1e6:.0f}MB，依赖走 tuna) ===")
    for attempt in range(1, 5):
        rc, out = sh(f"{PIP} install '{wheel}' -i {INDEX}")
        log(out[-800:])
        if rc == 0:
            log(f"torch install OK (attempt {attempt})")
            break
        log(f"torch attempt {attempt} failed rc={rc}，清理后重试")
        sh("pkill -f 'pip install'; sleep 3")
    else:
        log("FINAL: FAIL (torch)")
        sys.exit(1)

    log("=== 3. 安装 requirements 其余依赖 ===")
    for attempt in range(1, 5):
        rc, out = sh(f"{PIP} install -r {REQ} -i {INDEX}")
        log(out[-800:])
        if rc == 0:
            log(f"requirements OK (attempt {attempt})")
            break
        log(f"requirements attempt {attempt} failed rc={rc}，清理后重试")
        sh("pkill -f 'pip install'; sleep 3")
    else:
        log("FINAL: FAIL (requirements)")
        sys.exit(1)

    log("=== 4. 验证 ===")
    rc, out = sh("/root/miniconda3/bin/python -c "
                 "'import torch, peft, torchaudio, transformers; "
                 "print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), "
                 "\"peft\", peft.__version__, \"tf\", transformers.__version__)'")
    log("VERIFY:", out.strip())
    if rc == 0 and "cuda True" in out:
        log("FINAL: VERIFY_OK")
    else:
        log("FINAL: FAIL (verify)")
        sys.exit(1)


if __name__ == "__main__":
    main()
