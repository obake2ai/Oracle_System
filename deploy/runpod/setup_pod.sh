#!/usr/bin/env bash
# RunPod Pod の初回セットアップ (IAMAS 2026 / Organs 配信)
# 前提: テンプレートは runpod/pytorch (CUDA 12.x)、Volume を /workspace にマウント、HTTP ポート 8000 を公開
# 使い方: bash /workspace/Oracle_System/deploy/runpod/setup_pod.sh
set -euo pipefail

ROOT=/workspace/Oracle_System
cd "$ROOT"

echo "== apt: ffmpeg =="
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq ffmpeg fonts-noto-cjk >/dev/null
fi
ffmpeg -hide_banner -version | head -1

echo "== pip =="
python -m pip install -q -r requirements-stream.txt

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY

echo "== 代用モデル (重みが届くまでの動作確認用) =="
mkdir -p models
STANDIN=models/standin-stylegan3-r-afhqv2-512x512.pkl
if [ ! -f "$STANDIN" ]; then
  # NVIDIA 公式の学習済み StyleGAN3-R (研究・評価用ライセンス。展示には使わない)
  curl -L --fail -o "$STANDIN" \
    https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
fi
ls -la models/

echo "== 字幕フォント (代用: Noto。本番フォントはライセンス確認後) =="
mkdir -p data/fonts
ls /usr/share/fonts/opentype/noto/ 2>/dev/null | grep -i -E "NotoSansCJK-Regular" | head -2 || true

echo "セットアップ完了。起動は deploy/runpod/start_stream.sh"
