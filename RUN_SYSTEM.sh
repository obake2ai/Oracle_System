#!/bin/bash

# 1) 移動して git pull
cd /home/oracle/Oracle_System
git pull

# 2) 仮想環境をアクティベート
source venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.6"

# 3) pythonスクリプトをバックグラウンドで実行
python3 run_oracle_parallel.py &

# 4) 19:30 になったら強制的に止める（ループで待ち続ける例）
while true; do
  # 現在時刻を「HH:MM」形式で取得
  current_time=$(date +%H:%M)

  if [ "$current_time" = "19:30" ]; then
    # run_oracle_parallel.py を停止
    pkill -f run_oracle_parallel.py
    # ループを抜ける
    break
  fi

  # 1分ごとに判定
  sleep 60
done
