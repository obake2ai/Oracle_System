#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ubuntu（2 GPU搭載）環境において、StyleGAN3 によるリアルタイム映像生成と
GPT によるテキスト生成＋ChatGPT API 翻訳を組み合わせ、
映像上に英語（上半分）と日本語訳（下半分）の二段組でオーバーレイ表示するサンプルコードです。

※StyleGAN3 用の各種設定は、config/config.py の STYLEGAN_CONFIG をデフォルト値として利用し、
   CLI で上書き可能です。
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
import openai

# API キーは config/api.py からインポート（例: OPENAI_API_KEY）
from config.api import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# StyleGAN3 用の設定を config/config.py からインポート
from config.config import STYLEGAN_CONFIG

# src/realtime_generate.py 内の必要関数（循環参照にならないよう注意）
from src.realtime_generate import infinite_latent_smooth, infinite_latent_random_walk, img_resize_for_cv2
# GPT 用ローカルモデル（util/llm.py の GPT クラス）
from util.llm import GPT

# グローバル変数（描画テキスト：英語原文と日本語訳）
current_text = {"en": "", "ja": ""}
text_lock = threading.Lock()


# ──────────────────────────────
# ChatGPT API を利用した翻訳関数
def translate_to_japanese(text):
    prompt = f"次の英語のテキストを、なるべく元の文体を保ったまま、神秘的な神話テキストとして日本語に翻訳してください:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 必要に応じて変更してください
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        translated = response["choices"][0]["message"]["content"].strip()
        return translated
    except Exception as e:
        print("翻訳エラー:", e)
        return "翻訳エラー"


# ──────────────────────────────
# 自動改行（単語単位で）のヘルパー関数
def wrap_text(text, max_width, font, font_scale, thickness):
    """
    与えられたテキストを、cv2.getTextSize() を用いて max_width (ピクセル) を超えないように
    単語単位で改行し、行のリストを返します。
    単一の単語が max_width を超える場合は文字単位で改行します。
    """
    if not text:
        return []

    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        # すでに文字がある場合は半角スペースを追加
        test_line = current_line + (" " if current_line else "") + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if line_width <= max_width:
            current_line = test_line
        else:
            # 現在の行を確定
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # 単一の単語が max_width を超える場合：文字単位で改行
                sub_line = ""
                for char in word:
                    test_sub_line = sub_line + char
                    (sub_line_width, _), _ = cv2.getTextSize(test_sub_line, font, font_scale, thickness)
                    if sub_line_width <= max_width:
                        sub_line = test_sub_line
                    else:
                        lines.append(sub_line)
                        sub_line = char
                current_line = sub_line
    if current_line:
        lines.append(current_line)
    return lines


# ──────────────────────────────
# GPT による英語テキスト生成＋翻訳スレッド
def gpt_text_generator(model_path, prompt, max_new_tokens, context_length,
                       device, display_time=5.0, clear_time=0.5):
    global current_text

    tokenizer = tiktoken.get_encoding('gpt2')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {key.replace("_orig_mod.", ""): value
                  for key, value in checkpoint['model_state_dict'].items()}
    config = checkpoint['config']

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
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

        if isinstance(generated, str):
            en_text = generated
        else:
            if isinstance(generated, torch.Tensor):
                generated = generated.tolist()
            if isinstance(generated, (list, tuple)) and all(isinstance(token, int) for token in generated):
                en_text = tokenizer.decode(generated)
            else:
                en_text = str(generated)

        ja_text = translate_to_japanese(en_text)

        with text_lock:
            current_text = {"en": en_text, "ja": ja_text}

        time.sleep(display_time)
        with text_lock:
            current_text = {"en": "", "ja": ""}
        time.sleep(clear_time)


# ──────────────────────────────
# StyleGAN3 によるフレーム生成スレッド
def stylegan_frame_generator(frame_queue, stop_event, config_args):
    device = torch.device(config_args["stylegan_gpu"])
    noise_seed = config_args["noise_seed"]
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    os.makedirs(config_args["out_dir"], exist_ok=True)

    # STYLEGAN_CONFIG の全内容を Gs_kwargs に引き継ぐ
    Gs_kwargs = dnnlib.EasyDict(config_args)
    # キー "nXY" は内部的に変換して "countHW" としてセット
    if "nXY" in Gs_kwargs:
        nxy = Gs_kwargs.pop("nXY")
        Gs_kwargs.countHW = [int(s) for s in nxy.split('-')][::-1]

    model_path = config_args["model"]
    pkl_name = os.path.splitext(model_path)[0]
    custom = False if '.pkl' in model_path.lower() else True
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        rot = True if ('-r-' in model_path.lower() or 'sg3r-' in model_path.lower()) else False
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim
    label = torch.zeros([1, c_dim], device=device) if c_dim > 0 else None

    if config_args["method"] == "random_walk":
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=noise_seed, step_size=0.02)
    else:
        latent_gen = infinite_latent_smooth(z_dim=z_dim, device=device,
                                            cubic=config_args["cubic"],
                                            gauss=config_args["gauss"],
                                            seed=noise_seed,
                                            chunk_size=60,
                                            uniform=False)
    while not stop_event.is_set():
        z_current = next(latent_gen)
        with torch.no_grad():
            if custom and hasattr(Gs.synthesis, 'input'):
                trans_param = (torch.zeros([1, 2], device=device),
                               torch.zeros([1, 1], device=device),
                               torch.ones([1, 2], device=device))
                out = Gs(z_current, label, None, trans_param, torch.zeros(1, device=device),
                         truncation_psi=config_args["trunc"], noise_mode='const')
            else:
                out = Gs(z_current, label, None, truncation_psi=config_args["trunc"], noise_mode='const')
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_np = out[0].cpu().numpy()[..., ::-1]
        frame_queue.put(out_np)


# ──────────────────────────────
# 二段組（上：英語、下：日本語）のテキストオーバーレイ描画
def overlay_text_on_frame(frame, texts, font_scale=1.0, thickness=2,
                          font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255)):
    frame_h, frame_w = frame.shape[:2]
    max_text_width = int(frame_w * 0.9)

    en_lines = wrap_text(texts.get("en", ""), max_text_width, font, font_scale, thickness)
    ja_lines = wrap_text(texts.get("ja", ""), max_text_width, font, font_scale, thickness)

    gap = 5
    def get_block_height(lines):
        h_total = 0
        for line in lines:
            (_, line_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            h_total += line_h + gap
        return h_total - gap if lines else 0

    en_block_h = get_block_height(en_lines)
    ja_block_h = get_block_height(ja_lines)

    top_region_center = frame_h // 4
    bottom_region_center = frame_h * 3 // 4

    en_y0 = top_region_center - en_block_h // 2
    ja_y0 = bottom_region_center - ja_block_h // 2

    def draw_lines(lines, y0):
        y = y0
        for line in lines:
            (line_w, line_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = (frame_w - line_w) // 2
            cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_h + gap

    draw_lines(en_lines, en_y0)
    draw_lines(ja_lines, ja_y0)
    return frame


# ──────────────────────────────
# Click オプション（StyleGAN3 用パラメータは config/config.py の STYLEGAN_CONFIG をデフォルト値に）
@click.command()
@click.option('--out-dir', type=str, default=STYLEGAN_CONFIG['out_dir'], help="output directory")
@click.option('--model', type=str, default=STYLEGAN_CONFIG['model'], help="path to pkl checkpoint file")
@click.option('--labels', type=int, default=STYLEGAN_CONFIG['labels'], help="labels/categories for conditioning")
@click.option('--size', type=str, default=STYLEGAN_CONFIG['size'], help="Output resolution (e.g., 1280-720)")
@click.option('--scale-type', type=str, default=STYLEGAN_CONFIG['scale_type'], help="Scale type")
@click.option('--latmask', type=str, default=STYLEGAN_CONFIG['latmask'], help="external mask file or directory")
@click.option('--nxy', type=str, default=STYLEGAN_CONFIG['nXY'], help="multi latent frame split count by X and Y")
@click.option('--splitfine', type=float, default=STYLEGAN_CONFIG['splitfine'], help="split edge sharpness")
@click.option('--splitmax', type=int, default=STYLEGAN_CONFIG['splitmax'], help="max count of latents for frame splits")
@click.option('--trunc', type=float, default=STYLEGAN_CONFIG['trunc'], help="truncation psi")
@click.option('--save_lat', is_flag=True, default=STYLEGAN_CONFIG['save_lat'], help="save latent vectors to file")
@click.option('--verbose', is_flag=True, default=STYLEGAN_CONFIG['verbose'], help="verbose output")
@click.option('--noise_seed', type=int, default=STYLEGAN_CONFIG['noise_seed'], help="noise seed")
# Animation 関連
@click.option('--frames', type=str, default=STYLEGAN_CONFIG['frames'], help="total frames and interpolation step (e.g., 200-25)")
@click.option('--cubic', is_flag=True, default=STYLEGAN_CONFIG['cubic'], help="use cubic splines for smoothing")
@click.option('--gauss', is_flag=True, default=STYLEGAN_CONFIG['gauss'], help="use Gaussian smoothing")
# Transform SG3 関連
@click.option('--anim_trans', is_flag=True, default=STYLEGAN_CONFIG['anim_trans'], help="add translation animation")
@click.option('--anim_rot', is_flag=True, default=STYLEGAN_CONFIG['anim_rot'], help="add rotation animation")
@click.option('--shiftbase', type=float, default=STYLEGAN_CONFIG['shiftbase'], help="shift to the tile center")
@click.option('--shiftmax', type=float, default=STYLEGAN_CONFIG['shiftmax'], help="random walk around tile center")
@click.option('--digress', type=float, default=STYLEGAN_CONFIG['digress'], help="distortion strength")
# Affine Conversion
@click.option('--affine_scale', type=str, default=STYLEGAN_CONFIG['affine_scale'], help="affine scale (e.g., 1.0-1.0)")
# Video setting
@click.option('--framerate', type=int, default=STYLEGAN_CONFIG['framerate'], help="frame rate")
@click.option('--prores', is_flag=True, default=STYLEGAN_CONFIG['prores'], help="output video in ProRes format")
@click.option('--variations', type=int, default=STYLEGAN_CONFIG['variations'], help="number of variations")
# 無限生成方式
@click.option('--method', type=click.Choice(["smooth", "random_walk"]), default=STYLEGAN_CONFIG['method'], help="infinite realtime generation method")
# GPT 用オプション
@click.option('--gpt-model', type=str, default="./models/gpt_model_epoch_16000.pth", help="GPT model checkpoint path")
@click.option('--gpt-prompt', type=str, default="I'm praying: ", help="GPT generation prompt")
@click.option('--max-new-tokens', type=int, default=50, help="maximum new tokens for GPT")
@click.option('--context-length', type=int, default=512, help="GPT context length")
@click.option('--gpt-gpu', type=str, default="cuda:1", help="GPU for GPT")
@click.option('--display-time', type=float, default=5.0, help="display time for generated text (seconds)")
@click.option('--clear-time', type=float, default=0.5, help="clear time for text (seconds)")
def cli(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc,
        save_lat, verbose, noise_seed, frames, cubic, gauss, anim_trans, anim_rot, shiftbase,
        shiftmax, digress, affine_scale, framerate, prores, variations, method,
        gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_gpu, display_time, clear_time):
    """
    StyleGAN3 によるリアルタイム映像生成と GPT によるテキスト生成＋ChatGPT API 翻訳を組み合わせ、
    映像上に英語（上半分）と日本語訳（下半分）でオーバーレイ表示します。
    """
    # size, affine_scale の文字列をパース（例："1280-720" → [1280,720]、"1.0-1.0" → [1.0, 1.0]）
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

    try:
        if "-" in affine_scale:
            a, b = affine_scale.split("-")
            affine_parsed = [float(a), float(b)]
        else:
            affine_parsed = [1.0, 1.0]
    except Exception as e:
        print("affine_scale のパースに失敗しました。例: 1.0-1.0 の形式で指定してください。")
        raise e

    # CLI の値で上書きした内容をもとに、StyleGAN3 用設定辞書を作成
    config_args = {
        "out_dir": out_dir,
        "model": model,
        "labels": labels,
        "size": size_parsed,
        "scale_type": scale_type,
        "latmask": latmask,
        "nXY": nxy,
        "splitfine": splitfine,
        "splitmax": splitmax,
        "trunc": trunc,
        "save_lat": save_lat,
        "verbose": verbose,
        "noise_seed": noise_seed,
        "frames": frames,
        "cubic": cubic,
        "gauss": gauss,
        "anim_trans": anim_trans,
        "anim_rot": anim_rot,
        "shiftbase": shiftbase,
        "shiftmax": shiftmax,
        "digress": digress,
        "affine_scale": affine_parsed,
        "framerate": framerate,
        "prores": prores,
        "variations": variations,
        "method": method,
        "stylegan_gpu": "cuda:0"  # StyleGAN3 は固定で GPU0 を利用（必要に応じて変更）
    }

    frame_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    gan_thread = threading.Thread(target=stylegan_frame_generator,
                                  args=(frame_queue, stop_event, config_args),
                                  daemon=True)
    gan_thread.start()

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
            texts = current_text.copy()
        frame_with_text = overlay_text_on_frame(frame.copy(), texts, font_scale=1.0, thickness=2)
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
