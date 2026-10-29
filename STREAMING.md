# 《Organs》配信モード (IAMAS 2026)

2025 年の《Oracle Womb》展では、会場のローカル GPU 機 (Windows + WSL) が映像を生成し、画面に直接全画面表示していた。
IAMAS 展 (2026-12、大垣) では、**RunPod の GPU で生成し、HLS で会場へ配信する**。会場には小型 PC とプロジェクタだけを置く。

ブランチ: `iamas-2026-stream` (2025-03-15 の手元コミット 77d08ac の上。GitHub には未 push)

## 構成

```
RunPod Pod (GPU)                                  会場 (IAMAS Gallery 2)
run_oracle_parallel.py --output hls               小型 PC の Chrome (キオスク)
  StyleGAN3 胎児モデル + 祈りの GPT + 字幕          player/index.html
  -> src/stream_sink.py (一定 fps で ffmpeg へ)       ライブが止まったら fallback.mp4 に切替
  -> stream/stream.m3u8 + seg_*.ts                  再接続して安定したらライブに戻す
serve_hls.py :8000  --HTTPS (RunPod プロキシ)-->    -> プロジェクタ
```

- 従来の画面表示 (`--output screen`、既定) はそのまま残してある
- 生成が落ちたら黒を流し続けず、プロセスを終了する (start_stream.sh のループが 5 秒後に再起動)。モデル読み込み中は配信を始めないので、会場は予備映像のまま
- 配信の遅延は 10 秒前後。観客と対話しない作品なので問題にならない

## 追加・変更したもの

| ファイル | 内容 |
|---|---|
| `run_oracle_parallel.py` | `--output hls` ほか配信用オプション、headless 対応 (画面解像度の検出を画面モードだけに)、API キーは環境変数優先、`--text-source placeholder` (重み無しの動作確認)、生成停止の検知、`_out` の作成漏れ修正 |
| `src/stream_sink.py` | HLS 書き出し (一定 fps、最新フレームを繰り返して実時間を保つ、ffmpeg が落ちたら再起動) |
| `serve_hls.py` | HLS の配信と `/health` (CORS 付き、プレイリストは非キャッシュ) |
| `player/` | 会場プレイヤー (hls.js 1.5.13 同梱、予備映像への自動切替、`?debug=1` で状態表示) |
| `tools/synthetic_source.py` | GPU 無しで配信経路を試す合成映像 |
| `deploy/runpod/` | Pod の初回セットアップと起動・停止 |
| `src/torch_utils/ops/*_gradfix.py` | 新しい setuptools で消えた `pkg_resources` の代わりに `packaging` を使う |
| `requirements-stream.txt` | 配信モードの依存 |

## 試験の記録

- 2026-10-29 Mac (CPU) で確認済み
  - 合成映像 -> HLS -> serve_hls -> player: ライブ再生、配信停止から約 13 秒で予備映像へ、配信復帰後にライブへ自動で戻る (Chrome headless で 3 段階を撮影)
  - 本物の生成スクリプト `--output hls` + 代用モデル (NVIDIA 公式 StyleGAN3-R afhqv2 512) + 代用字幕: 1280x720 / 30fps の H.264 で配信され、日英字幕が重なることを確認
- 未確認: RunPod の GPU 上での速度、胎児モデルと祈りの GPT の重み (旧 PC から回収待ち)、本番フォント

## RunPod での手順

1. 個人の RunPod アカウントで Pod を作る (電通 KESHIKI の RunPod は使わない)
   - テンプレート: RunPod PyTorch (CUDA 12.x)。GPU: RTX 3090 / 4090 / A5000 などを機種指定 (Blackwell は torch cu128 が要る)
   - Volume を `/workspace` に、HTTP ポート `8000` を公開
   - 環境変数 `OPENAI_API_KEY` (和訳用、新しく発行したキー)
2. Pod のターミナルで
   ```bash
   cd /workspace && git clone -b iamas-2026-stream https://github.com/obake2ai/Oracle_System.git
   bash Oracle_System/deploy/runpod/setup_pod.sh
   bash Oracle_System/deploy/runpod/start_stream.sh          # 代用モデル + 代用字幕
   curl -s localhost:8000/health
   ```
3. 公開 URL `https://<POD_ID>-8000.proxy.runpod.net/stream.m3u8` を `player/config.js` の `streamUrl` に書く
4. 重みが届いたら `models/` に置いて
   ```bash
   MODEL=models/embryo-stylegan3-r-network-snapshot-000096 TEXT_SOURCE=gpt \
     bash deploy/runpod/start_stream.sh
   ```
   (`MODEL` は拡張子 `.pkl` を付けない。旧設定どおり独自ローダーで読む)

## 会場プレイヤー

```bash
# 小型 PC (Mac / Windows / Linux いずれも Chrome で可)
google-chrome --kiosk --autoplay-policy=no-user-gesture-required \
  --allow-file-access-from-files file:///path/to/player/index.html
```

- `player/fallback.mp4` に、同じシステムで事前に録った長尺映像を置く (Git には入れない)
- 状態確認: URL に `?debug=1` を付ける

## 未決

- **本番フォント**: 旧設定は FOT-筑紫ゴシック Pr5 D と Acumin (どちらも商用、Git から除外済み)。クラウドのサーバーにフォントファイルを置けるかはライセンス次第。確認できるまでは Noto で代用
- 生成解像度とスクリーン比: 旧 B1F は 2058x920 (約 2.2:1) で 8fps。IAMAS のスクリーンに合わせて決める
- 帯域: 8 Mbps で 1 時間あたり約 3.6 GB。会場回線と相談してビットレートを決める (`BITRATE=4M` 等)
