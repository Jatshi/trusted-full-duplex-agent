import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from lab import MemoryLab, sha
ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
lab = MemoryLab(ROOT/'assets.json')


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, kind='application/json'):
        raw = data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind+'; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/health':
            self.reply(200,lab.health)
        elif self.path == '/':
            self.reply(200,(ROOT/'panel.html').read_bytes(),'text/html')
        else:
            self.reply(404,{'error':'not_found'})

    def do_POST(self):
        if self.path != '/predict':
            self.reply(404,{'error':'not_found'}); return
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 16384:
                raise ValueError('request size')
            payload = json.loads(self.rfile.read(size))
            if set(payload) != {'controls'}:
                raise ValueError('controls only')
            result = lab.predict(payload['controls'])
            print(json.dumps({'event':'memory_lab_prediction','observed_events':result['observed_events'],
                              'predictions':result['predictions'],'rule':result['rule']},ensure_ascii=False),flush=True)
            self.reply(200,result)
        except (ValueError,KeyError,TypeError) as exc:
            self.reply(400,{'error':str(exc)})


print(json.dumps(lab.health),flush=True)
HTTPServer(('127.0.0.1',22681),Handler).serve_forever()
