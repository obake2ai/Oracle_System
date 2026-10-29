#!/usr/bin/env bash
# Organs の生成と HLS 配信を起動する (RunPod Pod 上)
#   MODEL       StyleGAN3 の重み (拡張子 .pkl を付けない。'-r-' を含む名前で回転付きとして読む)
#   TEXT_SOURCE gpt | placeholder (GPT の重みが無い間は placeholder)
#   OPENAI_API_KEY は Pod の環境変数で渡す (ファイルに書かない)
# 止める: bash deploy/runpod/start_stream.sh stop
set -uo pipefail
ROOT=/workspace/Oracle_System
cd "$ROOT"
LOG=$ROOT/log/stream
mkdir -p "$LOG"

if [ "${1:-}" = "stop" ]; then
  pkill -f serve_hls.py || true
  pkill -f run_oracle_parallel.py || true
  pkill -f "ffmpeg.*stream.m3u8" || true
  echo stopped; exit 0
fi

MODEL=${MODEL:-models/standin-stylegan3-r-afhqv2-512x512}
TEXT_SOURCE=${TEXT_SOURCE:-placeholder}
GEN_SIZE=${GEN_SIZE:-1610-720}        # 生成解像度 (旧 B1F は 2058-920 で 8fps、1610-720 で 14fps)
OUT_SIZE=${OUT_SIZE:-1920x1080}       # 配信フレーム (スクリーン比に合わせて黒帯で収める)
FPS=${FPS:-30}
BITRATE=${BITRATE:-8M}
ENCODER=${ENCODER:-libx264}
FONT_JA=${FONT_JA:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}
FONT_EN=${FONT_EN:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}
TRANSLATE=--translate
[ "$TEXT_SOURCE" = "placeholder" ] && TRANSLATE=--no-translate

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(python -c "import torch;c=torch.cuda.get_device_capability(0);print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo 8.6)}

nohup python serve_hls.py --dir stream --port 8000 > "$LOG/serve.log" 2>&1 &

# 生成側が落ちても自動で立ち上げ直す
nohup bash -c "
while true; do
  python run_oracle_parallel.py --output hls --hls-dir stream \
    --model '$MODEL' --size '$GEN_SIZE' --out-size '$OUT_SIZE' \
    --stream-fps $FPS --hls-bitrate $BITRATE --hls-encoder $ENCODER \
    --text-source $TEXT_SOURCE $TRANSLATE --font-ja '$FONT_JA' --font-en '$FONT_EN'
  echo \"[\$(date)] generator exited (\$?), restarting in 5 s\"
  sleep 5
done" > "$LOG/generator.log" 2>&1 &

sleep 3
echo "started. health: curl -s localhost:8000/health"
echo "public:  https://\${RUNPOD_POD_ID:-POD_ID}-8000.proxy.runpod.net/stream.m3u8"
