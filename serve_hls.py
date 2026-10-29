#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Serve the HLS folder written by HLSSink (IAMAS 2026).

- CORS open, so the venue player (a local file in Chrome kiosk) can fetch it.
- Playlists are never cached; segments may be.
- /health returns JSON with the age of the playlist, for remote monitoring.

Usage: python serve_hls.py --dir stream --port 8000
"""

import argparse
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PLAYLIST = "stream.m3u8"


class HLSHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".m3u8": "application/vnd.apple.mpegurl",
        ".ts": "video/mp2t",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        if self.path.split("?")[0].endswith(".m3u8") or self.path.startswith("/health"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            playlist = os.path.join(self.directory, PLAYLIST)
            now = time.time()
            body = {"ok": False, "playlist_age_s": None, "segments": 0, "time": now}
            if os.path.exists(playlist):
                age = now - os.path.getmtime(playlist)
                segs = [n for n in os.listdir(self.directory) if n.endswith(".ts")]
                body.update(ok=age < 10, playlist_age_s=round(age, 1), segments=len(segs))
            data = json.dumps(body).encode()
            self.send_response(200 if body["ok"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        # keep the console quiet: only report errors
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="stream")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    handler = partial(HLSHandler, directory=os.path.abspath(args.dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {os.path.abspath(args.dir)} on http://{args.host}:{args.port}/{PLAYLIST}")
    server.serve_forever()


if __name__ == "__main__":
    main()
