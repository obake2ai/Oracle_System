# -*- coding: utf-8 -*-
"""
HLS output for headless / cloud runs (IAMAS 2026).

The generator pushes frames whenever it has them; a writer thread feeds ffmpeg
at a fixed frame rate (repeating the latest frame when the generator is slower),
so the stream always runs in real time and the player never starves.
ffmpeg writes a rolling HLS playlist into `out_dir`, served by serve_hls.py.
"""

import os
import shutil
import subprocess
import threading
import time

import numpy as np


class HLSSink:
    def __init__(self, out_dir, width, height, fps=30, segment_time=2, list_size=10,
                 encoder="libx264", bitrate="8M", playlist="stream.m3u8", verbose=False):
        self.out_dir = out_dir
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.segment_time = int(segment_time)
        self.list_size = int(list_size)
        self.encoder = encoder
        self.bitrate = bitrate
        self.playlist = playlist
        self.verbose = verbose

        self._latest = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # 最初の本物のフレームが来るまで書き出さない (モデル読み込み中に黒を配信しない。
        # プレイリストが無い間、会場プレイヤーは予備映像を流し続ける)
        self._has_frame = threading.Event()
        self._proc = None
        self.frames_written = 0
        self.restarts = 0

        self._prepare_dir()
        self._start_ffmpeg()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------
    def _prepare_dir(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for name in os.listdir(self.out_dir):
            if name.endswith((".ts", ".m3u8", ".tmp")):
                try:
                    os.remove(os.path.join(self.out_dir, name))
                except OSError:
                    pass

    def _ffmpeg_cmd(self):
        gop = self.fps * self.segment_time
        cmd = [
            shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
            "-an",
            "-c:v", self.encoder,
        ]
        if self.encoder == "libx264":
            cmd += ["-preset", "veryfast", "-tune", "zerolatency"]
        elif self.encoder.endswith("_nvenc"):
            cmd += ["-preset", "p4"]
        cmd += [
            "-b:v", self.bitrate, "-maxrate", self.bitrate, "-bufsize", self.bitrate,
            "-pix_fmt", "yuv420p",
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", str(self.segment_time),
            "-hls_list_size", str(self.list_size),
            "-hls_flags", "delete_segments+independent_segments+temp_file+omit_endlist",
            "-hls_segment_filename", os.path.join(self.out_dir, "seg_%06d.ts"),
            os.path.join(self.out_dir, self.playlist),
        ]
        return cmd

    def _start_ffmpeg(self):
        cmd = self._ffmpeg_cmd()
        if self.verbose:
            print("[HLSSink] " + " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _restart_ffmpeg(self):
        self.restarts += 1
        print(f"\n[HLSSink] ffmpeg を再起動します ({self.restarts} 回目)")
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.close()
            if self._proc:
                self._proc.kill()
                self._proc.wait(timeout=5)
        except Exception:
            pass
        time.sleep(1)
        self._start_ffmpeg()

    # ---------------------------------------------------------------
    def push(self, frame):
        """Store the newest frame (BGR uint8). Resized only if the size differs."""
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))
        with self._lock:
            self._latest = np.ascontiguousarray(frame, dtype=np.uint8)
        self._has_frame.set()

    def _writer_loop(self):
        while not self._has_frame.wait(timeout=0.5):
            if self._stop.is_set():
                return
        interval = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                data = self._latest.tobytes()
            try:
                self._proc.stdin.write(data)
                self.frames_written += 1
            except (BrokenPipeError, ValueError, OSError):
                self._restart_ffmpeg()
                next_t = time.monotonic()
                continue
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                # fell more than 1 s behind (e.g. machine stalled): resync instead of bursting
                next_t = time.monotonic()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.close()
            if self._proc:
                self._proc.wait(timeout=10)
        except Exception:
            if self._proc:
                self._proc.kill()
