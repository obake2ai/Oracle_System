#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ubuntu（2 GPU搭載）環境において、StyleGAN3 によるリアルタイム映像生成と
GPT によるテキスト生成を組み合わせたプレビューを行うサンプルコードです。

・StyleGAN3 は GPU（例: --stylegan-gpu "cuda:0"）で動作し、リアルタイムプレビューを行います。
・GPT は別の GPU（例: --gpt-gpu "cuda:1"）で動作し、テキストを生成・一定秒表示後に消去します。
・Click を用いて各種パラメータ（シード、出力サイズ、モデルパス、GPU指定など）をコマンドラインから指定可能にしています。
"""

import os
import sys
sys.path.append("/root/Share/Oracle_System/src")
import time
import threading
import queue
import random

import cv2
import numpy as np
import torch
import dnnlib
import legacy
import click
import tiktoken

# src/realtime_generate.py 内の必要な関数をインポート
from src.realtime_generate import infinite_latent_smooth, infinite_latent_random_walk, img_resize_for_cv2
# GPT 用のクラス（src/generate_llm.py 内の処理を参考）
from util.llm import GPT


# ── グローバル変数（映像上に表示するテキスト） ──
current_text = ""
text_lock = threading.Lock()


# ── GPT によるテキスト生成スレッド ──
def gpt_text_generator(model_path, prompt, max_new_tokens, context_length,
                       device, display_time=5.0, clear_time=0.5):
    global current_text

    # tiktoken を用いて GPT-2 のエンコーディングを取得
    tokenizer = tiktoken.get_encoding('gpt2')

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {key.replace("_orig_mod.", ""): value
                  for key, value in checkpoint['model_state_dict'].items()}
    config = checkpoint['config']

    # GPT モデルの初期化
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        context_length=context_length,
        tokenizer=tokenizer
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        # プロンプトのエンコード
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

        # 生成結果の型によって分岐
        if isinstance(generated, str):
            # すでに文字列の場合はそのまま出力
            output_text = generated
        else:
            # もしテンソルならリストに変換
            if isinstance(generated, torch.Tensor):
                generated = generated.tolist()
            # もし整数のリストであれば、デコードを行う
            if isinstance(generated, (list, tuple)) and all(isinstance(token, int) for token in generated):
                output_text = tokenizer.decode(generated)
            else:
                # 万が一、想定外の型の場合は文字列化
                output_text = str(generated)

        with text_lock:
            current_text = output_text

        time.sleep(display_time)
        with text_lock:
            current_text = ""
        time.sleep(clear_time)

# ── StyleGAN3 によるフレーム生成スレッド ──
def stylegan_frame_generator(frame_queue, stop_event, config_args):
    """
    StyleGAN3 のネットワークを指定の GPU 上で読み込み、
    無限に潜在空間補間を行いながらフレームを生成して frame_queue に格納します。

    config_args は以下のキーを含む dict として想定します：
      - noise_seed: int
      - out_dir: str
      - verbose: bool
      - size: [height, width]
      - scale_type: str
      - nXY: str (例 "1-1")
      - splitfine: float
      - model: str (StyleGAN3 の pkl ファイルのパス)
      - trunc: float
      - method: "smooth" または "random_walk"
      - cubic: bool
      - gauss: bool
      - stylegan_gpu: 使用する GPU デバイス（例: "cuda:0"）
    """
    device = torch.device(config_args.get("stylegan_gpu", "cuda:0"))
    noise_seed = config_args.get("noise_seed", 3025)
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    os.makedirs(config_args.get("out_dir", "_out"), exist_ok=True)

    # ネットワーク読み込み用パラメータ
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = config_args.get("verbose", False)
    Gs_kwargs.size = config_args.get("size", [720, 1280])
    Gs_kwargs.scale_type = config_args.get("scale_type", "pad")

    # latent blending 用の設定（今回は単一フレーム表示なので nXY のまま）
    nxy = config_args.get("nXY", "1-1")
    nHW = [int(s) for s in nxy.split('-')][::-1]
    Gs_kwargs.countHW = nHW
    Gs_kwargs.splitfine = config_args.get("splitfine", 0)

    model_path = config_args.get("model", "models/embryo-stylegan3-r-network-snapshot-000096")
    pkl_name = os.path.splitext(model_path)[0]
    # custom フラグの判定（簡易な例）
    custom = False if '.pkl' in model_path.lower() else True
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        rot = True if ('-r-' in model_path.lower() or 'sg3r-' in model_path.lower()) else False
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim
    # 条件付けラベル（必要に応じて）
    label = torch.zeros([1, c_dim], device=device) if c_dim > 0 else None

    # 潜在ベクトル生成モードの選択
    method = config_args.get("method", "smooth")
    if method == "random_walk":
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=noise_seed, step_size=0.02)
    else:
        latent_gen = infinite_latent_smooth(z_dim=z_dim, device=device,
                                            cubic=config_args.get("cubic", False),
                                            gauss=config_args.get("gauss", False),
                                            seed=noise_seed,
                                            chunk_size=60,
                                            uniform=False)
    while not stop_event.is_set():
        z_current = next(latent_gen)
        with torch.no_grad():
            if custom and hasattr(Gs.synthesis, 'input'):
                # 本来は trans_param や dconst なども指定すべきですが、ここでは簡易化のため固定値を使用
                trans_param = (torch.zeros([1, 2], device=device),
                               torch.zeros([1, 1], device=device),
                               torch.ones([1, 2], device=device))
                out = Gs(z_current, label, None, trans_param, torch.zeros(1, device=device),
                         truncation_psi=config_args.get("trunc", 0.9), noise_mode='const')
            else:
                out = Gs(z_current, label, None, truncation_psi=config_args.get("trunc", 0.9), noise_mode='const')
        # 出力テンソルを [H, W, C] の uint8 画像（BGR順）に変換
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_np = out[0].cpu().numpy()[..., ::-1]
        frame_queue.put(out_np)


# ── フレームにテキストを中央配置でオーバーレイする関数 ──
def overlay_text_on_frame(frame, text, font_scale=1.0, thickness=2,
                          font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255)):
    """
    frame に対して、複数行のテキストを中央に描画して返します。
    """
    if text == "":
        return frame
    lines = text.split("\n")
    sizes = []
    total_height = 0
    gap = 5  # 行間の隙間
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        sizes.append((w, h))
        total_height += h + gap
    total_height -= gap
    frame_h, frame_w = frame.shape[:2]
    y0 = (frame_h - total_height) // 2
    for i, line in enumerate(lines):
        (w, h) = sizes[i]
        x = (frame_w - w) // 2
        y = y0 + h + i * (h + gap)
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


# ── Click によるコマンドラインオプション設定 ──
@click.command()
@click.option('--noise-seed', type=int, default=3025, help="StyleGAN3 の乱数シード")
@click.option('--out-dir', type=str, default="_out", help="StyleGAN3 の出力ディレクトリ")
@click.option('--verbose', is_flag=True, help="StyleGAN3 の詳細ログを出力")
@click.option('--size', type=str, default="720x1280", help="出力サイズ（例: 720x1280、height x width）")
@click.option('--scale-type', type=str, default="pad", help="StyleGAN3 のスケールタイプ")
@click.option('--nxy', type=str, default="1-1", help="Multi latent 分割数（例: 1-1）")
@click.option('--splitfine', type=float, default=0.0, help="分割時のエッジシャープネス")
@click.option('--stylegan-model', type=str, default="models/embryo-stylegan3-r-network-snapshot-000096",
              help="StyleGAN3 のモデルパス")
@click.option('--trunc', type=float, default=0.9, help="StyleGAN3 の truncation psi")
@click.option('--method', type=click.Choice(["smooth", "random_walk"]), default="smooth",
              help="潜在補間モード")
@click.option('--cubic/--no-cubic', default=False, help="cubic smoothing を使用するか")
@click.option('--gauss/--no-gauss', default=False, help="Gaussian smoothing を使用するか")
@click.option('--stylegan-gpu', type=str, default="cuda:0", help="StyleGAN3 に使用する GPU")
# GPT 用のオプション
@click.option('--gpt-model', type=str, default="./models/gpt_model_epoch_16000.pth", help="GPT モデルのチェックポイントパス")
@click.option('--gpt-prompt', type=str, default="I'm praying: ", help="GPT の生成プロンプト")
@click.option('--max-new-tokens', type=int, default=50, help="GPT の最大生成トークン数")
@click.option('--context-length', type=int, default=512, help="GPT のコンテキスト長")
@click.option('--gpt-gpu', type=str, default="cuda:1", help="GPT に使用する GPU")
@click.option('--display-time', type=float, default=5.0, help="生成テキストの表示時間（秒）")
@click.option('--clear-time', type=float, default=0.5, help="テキスト消去の待ち時間（秒）")
def cli(noise_seed, out_dir, verbose, size, scale_type, nxy, splitfine, stylegan_model, trunc,
        method, cubic, gauss, stylegan_gpu, gpt_model, gpt_prompt, max_new_tokens, context_length,
        gpt_gpu, display_time, clear_time):
    """
    StyleGAN3 によるリアルタイム映像生成と GPT によるテキスト生成を組み合わせ、
    映像上に中央配置でテキストオーバーレイを行います。
    """
    # サイズ文字列を "heightxwidth" 形式としてパース
    try:
        if "x" in size:
            h, w = size.split("x")
        elif "X" in size:
            h, w = size.split("X")
        elif "," in size:
            h, w = size.split(",")
        else:
            raise ValueError("Invalid size format")
        size_parsed = [int(h), int(w)]
    except Exception as e:
        print("サイズのパースに失敗しました。例: 720x1280 の形式で指定してください。")
        raise e

    # StyleGAN3 用の設定辞書を構築
    config_args = {
        "noise_seed": noise_seed,
        "out_dir": out_dir,
        "verbose": verbose,
        "size": size_parsed,
        "scale_type": scale_type,
        "nXY": nxy,
        "splitfine": splitfine,
        "model": stylegan_model,
        "trunc": trunc,
        "method": method,
        "cubic": cubic,
        "gauss": gauss,
        "stylegan_gpu": stylegan_gpu,
    }

    # フレームキューと終了制御用イベントの作成
    frame_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    # StyleGAN3 のフレーム生成スレッド開始
    gan_thread = threading.Thread(target=stylegan_frame_generator,
                                  args=(frame_queue, stop_event, config_args),
                                  daemon=True)
    gan_thread.start()

    # GPT のテキスト生成スレッド開始（指定 GPU を利用）
    gpt_device = torch.device(gpt_gpu if torch.cuda.is_available() else "cpu")
    gpt_thread = threading.Thread(target=gpt_text_generator,
                                  args=(gpt_model, gpt_prompt, max_new_tokens, context_length,
                                        gpt_device, display_time, clear_time),
                                  daemon=True)
    gpt_thread.start()

    print("リアルタイムプレビュー開始（'q' キーで終了）")
    fps_count = 0
    t0 = time.time()

    while True:
        frame = frame_queue.get()
        with text_lock:
            text = current_text
        frame_with_text = overlay_text_on_frame(frame.copy(), text, font_scale=1.0, thickness=2)
        frame_with_text = img_resize_for_cv2(frame_with_text)
        cv2.imshow("StyleGAN3 + GPT Overlay", frame_with_text)

        fps_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"\r{fps_count / elapsed:.2f} fps", end="")
            sys.stdout.flush()
            t0 = time.time()
            fps_count = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n終了します。")
            stop_event.set()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    cli()
