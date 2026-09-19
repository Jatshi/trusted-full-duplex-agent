#!/usr/bin/env python3
"""Serve the replay as real static HTML and optionally open a public tunnel."""
from __future__ import annotations

import argparse
import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"


class ReplayHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

    def do_GET(self):  # noqa: N802
        clean_path = self.path.split("?", 1)[0]
        if clean_path == "/":
            self.path = "/full_duplex_demo.html"
        elif clean_path == "/healthz":
            body = json.dumps({"status":"ok","artifact":"full_duplex_demo.html"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    artifact = DEMO_DIR / "full_duplex_demo.html"
    if not artifact.exists():
        raise SystemExit(f"Missing {artifact}; run scripts/build_duplex_replay.py first")
    server = ThreadingHTTPServer((args.host,args.port), ReplayHandler)
    print(f"LOCAL_URL=http://127.0.0.1:{args.port}", flush=True)
    if args.share:
        from gradio.networking import setup_tunnel
        url = setup_tunnel("127.0.0.1",args.port,secrets.token_urlsafe(32),None,None)
        print(f"PUBLIC_URL={url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
