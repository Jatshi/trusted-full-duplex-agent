#!/usr/bin/env python3
"""快速并行下载 pip 依赖大包并安装（绕过 aliyun 单连接 ~0.7MB/s 限速）。

流程：停旧 pip -> dry-run 拿 URL 清单 -> 8 路分片并行下载(可续传) -> 本地装 -> 补装 -> 验证。
日志直接打到 stdout（由外层 nohup 重定向到 fastdl.log）。
"""
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

WHEEL_DIR = "/root/autodl-tmp/wheels"
REQ = "/root/autodl-tmp/trusted-full-duplex-agent/requirements_minicpmo.txt"
PIP = "/root/miniconda3/bin/pip"
# tuna 支持 PEP 658 元数据：dry-run 解析依赖不用整包下载，快得多
INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"


def log(*a):
    print(*a, flush=True)


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def get_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def dl_range(url, start, end, path, retries=6):
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r, open(path + ".tmp", "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    f.write(c)
            if os.path.getsize(path + ".tmp") == want:
                os.replace(path + ".tmp", path)
                return
        except Exception as e:
            log(f"  range {start}-{end} attempt{k}: {e}")
            time.sleep(3)
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def pdownload(url, out, nconn=8):
    size = get_size(url)
    if size < 16 * 1024 * 1024:
        for k in range(5):
            try:
                urllib.request.urlretrieve(url, out)
                log(f"  {os.path.basename(out)} {size/1e6:.1f}MB direct OK")
                return
            except Exception as e:
                log(f"  {os.path.basename(out)} attempt{k}: {e}")
                time.sleep(2)
        raise RuntimeError(f"direct download failed: {url}")
    part = size // nconn
    jobs = []
    for i in range(nconn):
        s = i * part
        e = (i + 1) * part - 1 if i < nconn - 1 else size - 1
        jobs.append((s, e, f"{out}.part{i}"))
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(nconn) as ex:
        futs = [ex.submit(dl_range, url, s, e, p) for s, e, p in jobs]
        for f in futs:
            f.result()
    with open(out, "wb") as o:
        for _, _, p in jobs:
            with open(p, "rb") as f:
                while True:
                    c = f.read(1 << 22)
                    if not c:
                        break
                    o.write(c)
            os.remove(p)
    assert os.path.getsize(out) == size, f"size mismatch {out}"
    log(f"  {os.path.basename(out)} {size/1e6:.1f}MB 8-way in {time.time()-t0:.0f}s "
        f"({size/1e6/max(time.time()-t0,1):.1f}MB/s)")


def main():
    os.makedirs(WHEEL_DIR, exist_ok=True)
    log("=== 1. 停掉一切 pip/curl 下载（含 pip_retry / finish_pip；fastdl 旧实例由启动器负责）===")
    sh("pkill -f pip_retry.sh; pkill -f finish_pip.sh; pkill -f 'pip install'; "
       "pkill -f 'curl -L'; sleep 2")
    sh("ps aux | grep -E 'pip install|finish_pip' | grep -v grep || echo ALL_CLEAN")
    log("=== 2. dry-run 解析依赖 URL ===")
    rc, out = sh(f"{PIP} install -r {REQ} --dry-run --quiet --report /tmp/pipreport.json -i {INDEX}")
    if rc != 0:
        log("dry-run FAILED:", out[-2000:])
        sys.exit(1)
    with open("/tmp/pipreport.json") as f:
        rep = json.load(f)
    items = rep.get("install", [])
    if not items:
        log("nothing to install?!")
        sys.exit(0)
    log(f"待装 {len(items)} 个包")
    for f in glob.glob(f"{WHEEL_DIR}/*"):
        os.remove(f)
    total = 0
    for it in items:
        url = it["download_info"]["url"]
        fn = url.split("/")[-1].split("#")[0]
        out = os.path.join(WHEEL_DIR, fn)
        size = get_size(url)
        total += size
        log(f"downloading {fn} ({size/1e6:.1f}MB)")
        try:
            pdownload(url, out)
        except Exception as e:
            log(f"FAIL {fn}: {e} （留给 pip 兜底）")
    log(f"=== 3. 安装本地包（总计 {total/1e6:.0f}MB）===")
    files = sorted(glob.glob(f"{WHEEL_DIR}/*"))
    if files:
        rc, out = sh(f"{PIP} install " + " ".join(f"'{w}'" for w in files))
        log(out[-1200:])
        if rc != 0:
            log("本地安装失败，看上方日志")
    log("=== 4. 补装 requirements 剩余 ===")
    rc, out = sh(f"{PIP} install -r {REQ} -i {INDEX}")
    log(out[-1200:])
    log("=== 5. 验证 ===")
    rc, out = sh("/root/miniconda3/bin/python -c "
                 "'import torch, peft, torchaudio, transformers; "
                 "print(\"VERIFY_OK torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), "
                 "\"tf\", transformers.__version__)'")
    log("FINAL:", out.strip())


if __name__ == "__main__":
    main()
