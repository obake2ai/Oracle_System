#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU もモデルも使わずに HLS 配信経路 (HLSSink -> serve_hls.py -> player) を試すための合成映像。
動くグラデーションと時刻を描くだけ。作品の映像ではない。

Usage: python tools/synthetic_source.py --out-dir stream --size 1280x720 --fps 30 --seconds 0
       (--seconds 0 で止めるまで流し続ける)
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.stream_sink import HLSSink  # noqa: E402


def make_frame(w, h, t, font):
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    r = np.broadcast_to(0.5 + 0.5 * np.sin(2 * np.pi * (x + 0.10 * t)), (h, w))
    g = np.broadcast_to(0.5 + 0.5 * np.sin(2 * np.pi * (y + 0.07 * t)), (h, w))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y + 0.05 * t))
    img = (np.stack([b * 0.6, g * 0.4, r * 0.8], axis=-1) * 255).astype(np.uint8)  # BGR
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    d.text((40, 40), f"SYNTHETIC TEST  {stamp}", fill=(255, 255, 255), font=font)
    return np.asarray(pil)[:, :, ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="stream")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gen-fps", type=float, default=8.0, help="生成側の速さ (旧 B1F 設定は約 8fps)")
    ap.add_argument("--seconds", type=float, default=0)
    args = ap.parse_args()

    w, h = [int(v) for v in args.size.lower().split("x")]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(h / 12))
    except Exception:
        font = ImageFont.load_default()

    sink = HLSSink(args.out_dir, w, h, fps=args.fps)
    print(f"synthetic -> {os.path.abspath(args.out_dir)}/stream.m3u8  {w}x{h} out {args.fps}fps / gen {args.gen_fps}fps")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if args.seconds and t > args.seconds:
                break
            sink.push(make_frame(w, h, t, font))
            time.sleep(1.0 / args.gen_fps)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        print(f"frames written: {sink.frames_written}, ffmpeg restarts: {sink.restarts}")


if __name__ == "__main__":
    main()
