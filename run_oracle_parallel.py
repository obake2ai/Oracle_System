#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
このコードは、StyleGAN3 によるリアルタイム映像生成と GPT によるテキスト生成＋翻訳を組み合わせ、
映像上に日本語（上部、半透明）と英語（下部、半透明）のテキストをオーバーレイ表示します。
また、--debug オプションを指定すると、cProfile によるボトルネックのプロファイリングを実施し、
終了時に累積時間が一定以上（ここでは 0.1 秒以上）の関数のみを表示します。
テキスト生成はバックグラウンドで連続して行われ、表示タイミングに合わせてキューから取り出されます。

さらに、#transformSG3 の機能として、-at (--anim_trans), -ar (--anim_rot), -sb (--shiftbase), -sm (--shiftmax)
を追加し、StyleGAN3 の平行移動・回転アニメーションを有効にできます。
"""

import os
import os.path as osp
import sys
sys.path.append("./src")
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
from PIL import Image, ImageDraw, ImageFont
import datetime
import gc
import subprocess
import cProfile, pstats, io  # プロファイリング用モジュール
import shutil
import re

# psutil（メモリ監視用）
try:
    import psutil
    process = psutil.Process(os.getpid())
except ImportError:
    psutil = None


def remove_garbled_characters(text: str) -> str:
    """
    文字列中のチェックボックスなどの文字化け（例: □, ■, ☐, ☑, ☒, �）を除去する。
    必要に応じてパターンを拡張してください。
    """
    pattern = r"[□■☐☑☒�]"
    return re.sub(pattern, "", text)


# -------------------------------
# xrandr の出力を利用して総解像度を取得する
# -------------------------------
def get_total_screen_resolution():
    try:
        output = subprocess.check_output("xrandr | grep 'Screen 0:'", shell=True).decode()
        parts = output.split(',')
        for part in parts:
            if "current" in part:
                tokens = part.strip().split()
                width = int(tokens[1])
                height = int(tokens[3])
                return width, height
    except Exception as e:
        print("xrandr から解像度取得エラー:", e)
        return None, None

actual_screen_width, actual_screen_height = get_total_screen_resolution()
if not actual_screen_width or not actual_screen_height:
    try:
        import tkinter as tk
        root = tk.Tk()
        actual_screen_width = root.winfo_screenwidth()
        actual_screen_height = root.winfo_screenheight()
        root.destroy()
    except Exception as e:
        actual_screen_width, actual_screen_height = 1920, 1080
        print("xrandr, tkinter どちらも利用できなかったため、デフォルトの解像度 1920x1080 を使用します。")
print(f"総解像度: {actual_screen_width}x{actual_screen_height}")

# --- ここはあなたのプロジェクト環境に合わせて設定 ---
from config.api import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

from config.config import STYLEGAN_CONFIG, GEN_CONFIG
from config.prompts import CHATGPT_PROMPTS
# utilgan.py など外部スクリプトで latent_anima() や infinite_latent_smooth(), infinite_latent_random_walk() を読み込む
from util.utilgan import latent_anima
from src.realtime_generate import infinite_latent_smooth, infinite_latent_random_walk, img_resize_for_cv2
from util.llm import GPT

# グローバル変数：テキスト生成用のキュー（最大200件保持）
text_queue = queue.Queue(maxsize=200)


# -------------------------------
# 各種ユーティリティ関数
# -------------------------------
def get_text_size(font, text):
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if get_text_size(font, test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # 単語が長すぎる場合はバラして改行
                sub_line = ""
                for char in word:
                    test_sub_line = sub_line + char
                    if get_text_size(font, test_sub_line)[0] <= max_width:
                        sub_line = test_sub_line
                    else:
                        lines.append(sub_line)
                        sub_line = char
                current_line = sub_line
    if current_line:
        lines.append(current_line)
    return lines

def translate_to_japanese(text):
    prompt = f"{CHATGPT_PROMPTS['translate']}\n\n{text}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # 必要に応じて変更
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print("翻訳エラー:", e)
        return "翻訳エラー"


# -------------------------------
# text_generation_worker:
# バックグラウンドで連続してテキストを生成し、キューに格納する
# -------------------------------
def text_generation_worker(gpt_model_path, gpt_prompt, max_new_tokens, context_length, gpt_device,
                           frame_width, font_scale, font_path_en, font_path_ja):
    tokenizer = tiktoken.get_encoding('gpt2')
    checkpoint = torch.load(gpt_model_path, map_location=gpt_device)
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    config = checkpoint['config']
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        context_length=context_length,
        tokenizer=tokenizer
    ).to(gpt_device)
    model.load_state_dict(state_dict)
    model.eval()

    font_size = int(32 * font_scale)
    try:
        font_en = ImageFont.truetype(font_path_en, font_size)
    except Exception as e:
        print("英語フォント読み込みエラー:", e)
        font_en = ImageFont.load_default()
    try:
        font_ja = ImageFont.truetype(font_path_ja, font_size)
    except Exception as e:
        print("日本語フォント読み込みエラー:", e)
        font_ja = ImageFont.load_default()

    max_text_width = int(frame_width * 0.9)

    while True:
        # GPT推論
        input_ids = torch.tensor(tokenizer.encode(gpt_prompt), dtype=torch.long, device=gpt_device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

        if isinstance(generated, torch.Tensor):
            generated = generated.tolist()
        if isinstance(generated, (list, tuple)) and all(isinstance(token, int) for token in generated):
            en_text = tokenizer.decode(generated)
        else:
            en_text = str(generated)

        if en_text.startswith(gpt_prompt):
            en_text = en_text[len(gpt_prompt):].strip()
        en_text = en_text.replace("\n", " ").strip()
        en_text = remove_garbled_characters(en_text)

        # 翻訳
        ja_text = translate_to_japanese(en_text)
        ja_text = ja_text.replace("\n", " ").strip()
        ja_text = remove_garbled_characters(ja_text)

        en_lines = wrap_text(en_text, font_en, max_text_width)
        ja_lines = wrap_text(ja_text, font_ja, max_text_width)

        text_line = {"en": en_lines, "ja": ja_lines}
        try:
            text_queue.put(text_line, timeout=5)
        except queue.Full:
            # キューが満杯なら少し待って再トライ
            time.sleep(1)


# -------------------------------
# blend_overlay: オーバーレイとフレームの合成
# -------------------------------
def blend_overlay(frame, overlay):
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA).astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    alpha = overlay_float[:, :, 3:4] / 255.0
    blended = frame_bgra.copy()
    blended[:, :, :3] = frame_bgra[:, :, :3] * (1 - alpha) + overlay_float[:, :, :3] * alpha
    blended_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    return blended_bgr


# -------------------------------
# create_text_overlay: 字幕オーバーレイ画像生成（透過度付き）
# -------------------------------
def create_text_overlay(frame_shape, texts, subtitle_ja_y, subtitle_en_y, thickness,
                        font_path_en, font_path_ja, font_size_ja, font_size_en):
    h, w = frame_shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    try:
        font_ja = ImageFont.truetype(font_path_ja, font_size_ja)
    except Exception as e:
        print("日本語フォント読み込みエラー:", e)
        font_ja = ImageFont.load_default()
    try:
        font_en = ImageFont.truetype(font_path_en, font_size_en)
    except Exception as e:
        print("英語フォント読み込みエラー:", e)
        font_en = ImageFont.load_default()

    ja_lines = texts.get("ja", [])
    en_lines = texts.get("en", [])

    y_ja = int(h * (subtitle_ja_y / 100.0))
    y_en = int(h * (subtitle_en_y / 100.0))

    draw = ImageDraw.Draw(overlay)
    text_fill = (255, 255, 255, 230)
    stroke_fill = (0, 0, 0, 230)

    for line in ja_lines:
        line_w, line_h = get_text_size(font_ja, line)
        x = (w - line_w) // 2
        draw.text((x, y_ja), line, font=font_ja, fill=text_fill,
                  stroke_width=thickness, stroke_fill=stroke_fill)
        y_ja += line_h

    for line in en_lines:
        line_w, line_h = get_text_size(font_en, line)
        x = (w - line_w) // 2
        draw.text((x, y_en), line, font=font_en, fill=text_fill,
                  stroke_width=thickness, stroke_fill=stroke_fill)
        y_en += line_h

    return np.array(overlay)


# -------------------------------
# letterbox_frame: 生成フレームを画面解像度に合わせる（上下を黒で埋める／クロップ）
# -------------------------------
def letterbox_frame(frame, target_width, target_height):
    orig_h, orig_w = frame.shape[:2]
    scale_factor = target_width / orig_w
    new_h = int(orig_h * scale_factor)
    resized_frame = cv2.resize(frame, (target_width, new_h))

    if new_h < target_height:
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        padded_frame = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, 0, 0,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_frame
    elif new_h > target_height:
        offset = (new_h - target_height) // 2
        cropped_frame = resized_frame[offset: offset + target_height, :]
        return cropped_frame
    else:
        return resized_frame


# -------------------------------
# stylegan_frame_generator:
# （メイン生成スレッド）
# -------------------------------
def stylegan_frame_generator(frame_queue, stop_event, config_args):
    """
    #transformSG3 のアニメーション機能を利用して、
    anim_trans, anim_rot, shiftbase, shiftmax を反映したフレームを連続生成。
    """
    device = torch.device(config_args["stylegan_gpu"])
    noise_seed = config_args["noise_seed"]
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    os.makedirs(config_args["out_dir"], exist_ok=True)

    Gs_kwargs = dnnlib.EasyDict()
    for key in ["verbose", "size", "scale_type"]:
        Gs_kwargs[key] = config_args[key]

    # latmask の処理
    if config_args["latmask"] is None:
        nxy = config_args["nXY"]
        nHW = [int(s) for s in nxy.split('-')][::-1]
        n_mult = nHW[0] * nHW[1]
        if config_args["splitmax"] is not None:
            n_mult = min(n_mult, config_args["splitmax"])
        if config_args["verbose"] and n_mult > 1:
            print(f"Latent blending with split frame {nHW[1]} x {nHW[0]}")
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = config_args["splitfine"]
        lmask = [None]
    else:
        # 外部マスク
        n_mult = 2
        nHW = [1, 1]
        if osp.isfile(config_args["latmask"]):
            # 単一ファイル
            from util.utilgan import img_read
            mask = img_read(config_args["latmask"])
            lmask = np.asarray([[mask[:, :, 0] / 255.]])  # shape [1,1,h,w]
        elif osp.isdir(config_args["latmask"]):
            # ディレクトリ -> 複数
            from util.utilgan import img_list, img_read
            files = img_list(config_args["latmask"])
            lmask = np.expand_dims(
                np.asarray([img_read(f)[:, :, 0] / 255. for f in files]),
                1
            )  # shape [N,1,h,w]
        else:
            print(' !! Blending mask not found:', config_args["latmask"])
            exit(1)

        if config_args["verbose"]:
            print(' Latent blending with mask', config_args["latmask"], lmask.shape)

        lmask = np.concatenate((lmask, 1 - lmask), 1)  # shape [N,2,h,w]
        lmask = torch.from_numpy(lmask).to(device)

    frames_val, fstep_val = [int(x) for x in config_args["frames"].split('-')]

    # StyleGAN3モデルの読み込み
    model_path = config_args["model"]
    pkl_name = osp.splitext(model_path)[0]
    custom = False if '.pkl' in model_path.lower() else True
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        rot = True if ('-r-' in model_path.lower() or 'sg3r-' in model_path.lower()) else False
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim

    # ラベル (c_dim > 0 の場合のみ)
    if c_dim > 0 and config_args["labels"] is not None:
        label = torch.zeros([1, c_dim], device=device)
        label_idx = min(int(config_args["labels"]), c_dim - 1)
        label[0, label_idx] = 1
    else:
        label = None

    # ---- #transformSG3: anim_trans, anim_rot, shiftbase, shiftmax ----
    if hasattr(Gs.synthesis, 'input'):
        # 平行移動
        if config_args["anim_trans"]:
            # nHW: [h_num, w_num], n_mult = h_num*w_num
            hw_centers = [np.linspace(-1 + 1/n, 1 - 1/n, n) for n in [nHW[0], nHW[1]]]
            yy, xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in config_args["size"]]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * np.array(xscale) \
                         * 0.5 * config_args["shiftbase"]
            hw_scales = np.array([2. / n for n in nHW]) * config_args["shiftmax"]
            # latent_anima => [frames_val, n_mult, 2]
            shifts = latent_anima((n_mult, 2), frames_val, fstep_val, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            # anim_trans 無効時はゼロシフト
            shifts = np.zeros((1, n_mult, 2))

        # 回転
        if config_args["anim_rot"]:
            # latent_anima => [frames_val, n_mult, 1]
            angles = latent_anima((n_mult, 1), frames_val, frames_val//4, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            angles = (angles - 0.5) * 180.  # -90度～+90度程度
        else:
            angles = np.zeros((1, n_mult, 1))

        # スケール
        scale_array = np.array(config_args["affine_scale"], dtype=np.float32)  # ex) [1.0,1.0]
        # アニメーションのフレーム数分同じスケール
        scales = np.tile(scale_array, (shifts.shape[0], shifts.shape[1], 1))

        # torchテンソル化
        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        scales = torch.from_numpy(scales).to(device)

        trans_params = list(zip(shifts, angles, scales))
    else:
        trans_params = None

    # distortion
    if hasattr(Gs.synthesis, 'input'):
        first_layer_channels = Gs.synthesis.input.channels
        first_layer_size = Gs.synthesis.input.size
        if isinstance(first_layer_size, (list, tuple, np.ndarray)):
            h_in, w_in = first_layer_size[0], first_layer_size[1]
        else:
            h_in, w_in = first_layer_size, first_layer_size
        shape_for_dconst = [1, first_layer_channels, h_in, w_in]
        if config_args["digress"] != 0:
            dconst_list = []
            # frames_val の長さだけアニメしてもよいが、とりあえず同じサイズに
            dconst_array = latent_anima(shape_for_dconst, frames_val, fstep_val,
                                        cubic=True, seed=noise_seed, verbose=False)
            # n_multぶん縦に連結 => [frames_val, n_mult*ch, h_in, w_in]
            for i in range(n_mult):
                dconst_list.append(config_args["digress"] * dconst_array)
            dconst = np.concatenate(dconst_list, axis=1)
        else:
            dconst = np.zeros([frames_val, 1, first_layer_channels, h_in, w_in], dtype=np.float32)
        dconst = torch.from_numpy(dconst).to(device).to(torch.float32)
    else:
        dconst = None

    # 1フレームウォームアップ
    with torch.no_grad():
        if custom and hasattr(Gs.synthesis, 'input'):
            dummy_trans_param = (torch.zeros([1,2], device=device),
                                 torch.zeros([1,1], device=device),
                                 torch.ones([1,2], device=device))
            _ = Gs(torch.randn([1, z_dim], device=device), label,
                   lmask[0] if lmask is not None else None,
                   dummy_trans_param,
                   dconst[0] if dconst is not None else None,
                   noise_mode='const')
        else:
            _ = Gs(torch.randn([1, z_dim], device=device), label,
                   truncation_psi=config_args["trunc"], noise_mode='const')

    frame_idx_local = 0
    frame_idx = 0

    # どの無限生成メソッドを使うか
    if config_args["method"] == "random_walk":
        print(" [Info] Using random_walk for latents.")
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=noise_seed, step_size=config_args["random_step_size"])
    else:
        print(" [Info] Using smooth for latents.")
        latent_gen = infinite_latent_smooth(
            z_dim=z_dim, device=device,
            cubic=config_args["cubic"], gauss=config_args["gauss"],
            seed=noise_seed,
            chunk_size=config_args["chunk_size"],
            uniform=False
        )

    while not stop_event.is_set():
        z_current = next(latent_gen)  # shape [1, z_dim]

        with torch.no_grad():
            if custom and hasattr(Gs.synthesis, 'input'):
                latmask = lmask[frame_idx_local % len(lmask)] if lmask is not None else None
                if dconst is not None and dconst.shape[0] > 1:
                    dconst_current = dconst[frame_idx % dconst.shape[0]]
                else:
                    dconst_current = None
                if trans_params is not None:
                    trans_param = trans_params[frame_idx % len(trans_params)]
                else:
                    trans_param = None

                out = Gs(z_current, label, latmask, trans_param, dconst_current,
                         truncation_psi=config_args["trunc"], noise_mode='const')
            else:
                # SG2 or fallback
                out = Gs(z_current, label,
                         truncation_psi=config_args["trunc"], noise_mode='const')

        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0,255).to(torch.uint8)
        out_np = out[0].cpu().numpy()[..., ::-1]  # RGB->BGR

        try:
            frame_queue.put(out_np, block=False)
        except queue.Full:
            # キューがフルなら古いフレームを捨てる
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(out_np, block=False)

        frame_idx_local += 1
        frame_idx += 1


# -------------------------------
# メインのCLI
# -------------------------------
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
@click.option('--frames', type=str, default=STYLEGAN_CONFIG['frames'], help="total frames and interpolation step (e.g., 200-25)")
@click.option('--cubic', is_flag=True, default=STYLEGAN_CONFIG['cubic'], help="use cubic splines for smoothing")
@click.option('--gauss', is_flag=True, default=STYLEGAN_CONFIG['gauss'], help="use Gaussian smoothing")

# ---- ここで短縮オプションを追加（#transformSG3相当）----
@click.option('-at', '--anim_trans', is_flag=True, default=STYLEGAN_CONFIG['anim_trans'],
              help="add translation animation")
@click.option('-ar', '--anim_rot', is_flag=True, default=STYLEGAN_CONFIG['anim_rot'],
              help="add rotation animation")
@click.option('-sb', '--shiftbase', type=float, default=STYLEGAN_CONFIG['shiftbase'],
              help="shift to the tile center")
@click.option('-sm', '--shiftmax', type=float, default=STYLEGAN_CONFIG['shiftmax'],
              help="random walk around tile center")

@click.option('--digress', type=float, default=STYLEGAN_CONFIG['digress'], help="distortion strength")
@click.option('--affine_scale', type=str, default=STYLEGAN_CONFIG['affine_scale'], help="affine scale (e.g., 1.0-1.0)")
@click.option('--framerate', type=int, default=STYLEGAN_CONFIG['framerate'], help="frame rate")
@click.option('--prores', is_flag=True, default=STYLEGAN_CONFIG['prores'], help="output video in ProRes format")
@click.option('--variations', type=int, default=STYLEGAN_CONFIG['variations'], help="number of variations")
@click.option('--method', type=click.Choice(["smooth", "random_walk"]), default=STYLEGAN_CONFIG['method'],
              help="infinite realtime generation method")
@click.option('--chunk_size', type=int, default=STYLEGAN_CONFIG['chunk_size'], help="step size for infinite generation (smooth)")
@click.option('--random_step_size', type=float, default=STYLEGAN_CONFIG['random_step_size'], help="step size for infinite generation (random)")
@click.option('--gpt-model', type=str, default=STYLEGAN_CONFIG['gpt_model'], help="GPT model checkpoint path")
@click.option('--gpt-prompt', type=str, default=STYLEGAN_CONFIG['gpt_prompt'], help="GPT generation prompt")
@click.option('--max-new-tokens', type=int, default=STYLEGAN_CONFIG['max_new_tokens'], help="maximum new tokens for GPT")
@click.option('--context-length', type=int, default=STYLEGAN_CONFIG['context_length'], help="GPT context length")
@click.option('--gpt-gpu', type=str, default=STYLEGAN_CONFIG['gpt_gpu'], help="GPU for GPT")
@click.option('--display-time', type=float, default=STYLEGAN_CONFIG['display_time'], help="display time for generated text (seconds)")
@click.option('--clear-time', type=float, default=STYLEGAN_CONFIG['clear_time'], help="clear time for text (seconds)")
@click.option('--font-scale', type=float, default=STYLEGAN_CONFIG['default_font_scale'], help="default font scale for overlay text")
@click.option('--font-thickness', type=int, default=STYLEGAN_CONFIG['default_font_thickness'],
              help="default font thickness for overlay text")
@click.option('--debug', is_flag=True, default=False, help="Enable profiling of bottlenecks")
@click.option('--fullscreen/--windowed', default=True,
              help="Use fullscreen mode if enabled; otherwise use window mode")
def cli(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc,
        save_lat, verbose, noise_seed, frames, cubic, gauss,
        anim_trans, anim_rot, shiftbase, shiftmax,
        digress, affine_scale, framerate, prores, variations, method, chunk_size, random_step_size,
        gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_gpu,
        display_time, clear_time, font_scale, font_thickness,
        sg_gpu=None, debug=False, fullscreen=True):
    """
    CLIエントリポイント。--debug オプションでプロファイリング。
    """
    profiler = None
    if debug:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # 解析: size
        try:
            if "-" in size:
                w, h = size.split("-")
            elif "x" in size.lower():
                w, h = size.lower().split("x")
            elif "X" in size:
                w, h = size.split("X")
            elif "," in size:
                w, h = size.split(",")
            else:
                raise ValueError("Invalid size format (例: 1280x720)")
            size_parsed = [int(h), int(w)]
        except Exception as e:
            print("サイズのパースに失敗しました。例: 720x1280 の形式で指定してください。")
            raise e

        # 解析: affine_scale
        try:
            if "-" in affine_scale:
                a_s1, a_s2 = affine_scale.split("-")
                affine_parsed = [float(a_s1), float(a_s2)]
            else:
                # デフォルト
                affine_parsed = [1.0, 1.0]
        except Exception as e:
            print("affine_scale のパースに失敗しました。例: 1.0-1.0 の形式で指定してください。")
            raise e

        # device (stylegan用): 旧引数名 sg_gpu が空のときは 'cuda' にするなど
        if sg_gpu is None:
            sg_gpu = STYLEGAN_CONFIG["sg_gpu"]  # デフォルト
        stylegan_device = sg_gpu if torch.cuda.is_available() else "cpu"

        # 設定まとめ
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
            "chunk_size": chunk_size,
            "random_step_size": random_step_size,
            "stylegan_gpu": stylegan_device,
            "font_scale": font_scale,
            "font_thickness": font_thickness,
        }

        # ログ出力用
        log_base = "log"
        os.makedirs(log_base, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_base, timestamp)
        os.makedirs(log_dir, exist_ok=True)

        # GPT用device
        gpt_device = torch.device(gpt_gpu if torch.cuda.is_available() else "cpu")

        # 背景でGPTテキスト生成ワーカーを起動
        text_thread = threading.Thread(
            target=text_generation_worker,
            args=(gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_device,
                  size_parsed[1],
                  font_scale, STYLEGAN_CONFIG['font_path_en'], STYLEGAN_CONFIG['font_path_ja']),
            daemon=True
        )
        text_thread.start()

        # 出力フォルダの整理例（必要に応じて）
        out_dirs = ['outputs/12x3-A', 'outputs/12x3-B']
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)

        # （サンプルとして）out_dirの中身を全消去
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"ファイル {file_path} の削除に失敗しました: {e}")

        screenshot_interval = GEN_CONFIG.get("generate_interval", 300)
        last_screenshot_time = time.time()

        # フレーム取得用
        frame_queue = queue.Queue(maxsize=30)
        stop_event = threading.Event()

        # StyleGAN3生成スレッド開始
        gan_thread = threading.Thread(
            target=stylegan_frame_generator,
            args=(frame_queue, stop_event, config_args),
            daemon=True
        )
        gan_thread.start()

        print("リアルタイムプレビュー開始（'q' キーで終了）")

        screen_width = actual_screen_width
        screen_height = actual_screen_height
        window_name = "Display"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        context_reinit_interval = 600
        last_context_reinit_time = time.time()
        last_gc_time = time.time()
        gc_interval = 60

        fps_count = 0
        t0 = time.time()

        # 初期フレームを取得（blocking）
        curr_frame = frame_queue.get()
        prev_frame = curr_frame.copy()
        last_frame_update = time.time()
        display_interval_ms = 33
        stylegan_interval = 0.1

        # テキスト表示用
        current_text = {"en": [], "ja": []}
        text_visible = True
        last_text_change = time.time()

        while True:
            now = time.time()
            # 定期的なGC
            if now - last_gc_time >= gc_interval:
                gc.collect()
                last_gc_time = now
                if psutil is not None:
                    mem = process.memory_info().rss / (1024 * 1024)
                    print(f"\n[GC] メモリ使用量: {mem:.2f} MB")

            # ウィンドウリサイズなどの再設定
            if now - last_context_reinit_time >= context_reinit_interval:
                print("\n[Context Reinit] OpenCV ウィンドウを再設定しました。")
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                last_context_reinit_time = now

            # フレームキューが溜まっている場合は最新を反映
            try:
                while True:
                    new_frame = frame_queue.get_nowait()
                    prev_frame = curr_frame
                    curr_frame = new_frame
                    last_frame_update = time.time()
            except queue.Empty:
                pass

            t = (time.time() - last_frame_update) / stylegan_interval
            if t > 1:
                t = 1.0

            # テキスト表示のON/OFF切り替え
            if text_visible and now - last_text_change >= display_time:
                text_visible = False
                last_text_change = now
            elif not text_visible and now - last_text_change >= clear_time:
                # 新しいキューがあれば取り出し、なければ空
                try:
                    current_text = text_queue.get_nowait()
                except queue.Empty:
                    current_text = {"en": [], "ja": []}
                text_visible = True
                last_text_change = now

            display_text = current_text if text_visible else {"en": [], "ja": []}

            # テキストオーバーレイ合成
            current_overlay = create_text_overlay(
                curr_frame.shape,
                display_text,
                STYLEGAN_CONFIG['subtitle_ja_y'],
                STYLEGAN_CONFIG['subtitle_en_y'],
                font_thickness,
                STYLEGAN_CONFIG['font_path_en'],
                STYLEGAN_CONFIG['font_path_ja'],
                STYLEGAN_CONFIG['subtitle_ja_font_size'],
                STYLEGAN_CONFIG['subtitle_en_font_size']
            )
            if current_overlay is not None:
                frame_with_text = blend_overlay(curr_frame, current_overlay)
            else:
                frame_with_text = curr_frame

            # 画面サイズに合わせる (letterbox)
            letterboxed_frame = letterbox_frame(frame_with_text, screen_width, screen_height)
            cv2.imshow(window_name, letterboxed_frame)

            # スクリーンショット保存タイミング
            if now - last_screenshot_time >= screenshot_interval:
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
                for save_dir in out_dirs:
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, letterboxed_frame)
                print(f"\n[INFO] Screenshot saved as {filename} in directories: {out_dirs}")
                last_screenshot_time = now

            # FPS計測
            fps_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                print(f"\r{fps_count / elapsed:.2f} fps", end="")
                t0 = time.time()
                fps_count = 0

            key = cv2.waitKey(display_interval_ms) & 0xFF
            if key == ord('q'):
                print("\n'q' が押されたため終了します。")
                stop_event.set()
                break

        cv2.destroyAllWindows()

    finally:
        if profiler is not None:
            profiler.disable()
            ps = pstats.Stats(profiler)
            ps.strip_dirs().sort_stats("cumtime")

            threshold = 0.1
            print("\n=== プロファイリング結果 (累積時間が {:.3f} 秒以上) ===".format(threshold))
            sorted_stats = sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True)
            for func, stat in sorted_stats:
                cc, nc, tt, ct, callers = stat
                if ct >= threshold:
                    func_desc = f"{func[0]}:{func[1]}({func[2]})"
                    print(f"{func_desc:60s}  Call Count: {nc:4d}  Total Time: {tt:8.6f}  Cumulative Time: {ct:8.6f}")

if __name__ == '__main__':
    cli()
