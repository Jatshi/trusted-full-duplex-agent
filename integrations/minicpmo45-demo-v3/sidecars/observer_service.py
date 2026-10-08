"""Local-only persistent real pretrained components, CPU2 threads; no training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np

def sha(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--assets",required=True)
    args=parser.parse_args()
    assets=Path(args.assets)
    source=assets/"easy_turn_src/Easy_Turn"
    sys.path.insert(0,str(source))
    example=source/"examples/wenetspeech/whisper"
    os.chdir(example)
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from wenet.llm_asr.init_llmasr import init_llmasr
    import librosa
    from tfd.turntaking.smart_turn_pretrained import SmartTurnV32, MODEL_SHA256
    from tfd.duplex_policy.demo_observers import parse_easy_turn, load_official_log_mel
    compute_log_mel_spectrogram = load_official_log_mel(source/"wenet/dataset/processor.py", torch, librosa)
    config=yaml.safe_load((example/"conf/train.yaml").read_text())
    config["llm_path"]=str(assets/"qwen2_5_0_5b")
    config["tokenizer_conf"]["llm_path"]=config["llm_path"]
    checkpoint=assets/"easy_turn/checkpoint.pt"
    assert checkpoint.stat().st_size == 3451801684
    model, loaded_config=init_llmasr(SimpleNamespace(checkpoint=str(checkpoint)),config)
    model=model.float().cpu().eval()
    model.max_length=100
    model.do_sample=False
    prompt=yaml.safe_load((example/"conf/prompt.yaml").read_text())["<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"][0]
    smart=SmartTurnV32(assets/"smart-turn-v3.2-cpu.onnx")
    asset_info={"smart_turn_sha256":MODEL_SHA256,"easy_turn_sha256":sha(checkpoint),"easy_turn_qwen_sha256":sha(assets/"qwen2_5_0_5b/model.safetensors"),"easy_turn_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"easy_turn_config_sha256":sha(example/"conf/train.yaml")}
    print(json.dumps({"ready":True,"device":"cpu","assets":asset_info}),flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def respond(self,code,value):
            payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(payload))); self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            if self.path != "/health": return self.respond(404,{"error":"unknown"})
            self.respond(200,{"status":"ready","device":"cpu","assets":asset_info})
        def do_POST(self):
            if self.path != "/predict": return self.respond(404,{"error":"unknown"})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0 < length <= 750000: raise ValueError("Body limit")
                body=json.loads(self.rfile.read(length))
                if body.get("sample_rate") != 16000: raise ValueError("16k required")
                wave=np.frombuffer(base64.b64decode(body["audio"],validate=True),dtype=np.float32).copy()
                if not 400 <= wave.size <= 128000 or not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.01:
                    raise ValueError("Normalized finite 0.025..8s mono required")
                start=time.monotonic()
                p=smart.probability_complete(wave)
                smart_ms=(time.monotonic()-start)*1000
                sample={"key":"live_prefix","label":[],"sample_rate":16000,"wav":torch.from_numpy(wave).unsqueeze(0)}
                feat=next(compute_log_mel_spectrogram(iter([sample])))["feat"].unsqueeze(0)
                start=time.monotonic()
                with torch.inference_mode():
                    raw=model.generate(wavs=feat,wavs_len=torch.tensor([feat.shape[1]]),prompt=prompt)
                result={"smart_turn_p_complete":p,"smart_turn_ms":smart_ms,"easy_turn_text":str(raw[0])[:2000],"easy_turn_label":parse_easy_turn(raw),"easy_turn_ms":(time.monotonic()-start)*1000,"device":"cpu","controls_output":False}
                print(json.dumps({"prediction":result},ensure_ascii=False),flush=True)
                self.respond(200,result)
            except Exception as exc:
                traceback.print_exc()
                self.respond(400,{"error":type(exc).__name__+": "+str(exc)[:160]})
    # Single prediction at a time; main Demo has its own max-one pending queue.
    HTTPServer(("127.0.0.1",22680),Handler).serve_forever()

if __name__ == "__main__":
    main()
