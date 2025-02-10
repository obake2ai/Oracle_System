#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ubuntu（2 GPU搭載）環境において、StyleGAN3 によるリアルタイム映像生成と
GPT によるテキスト生成＋ChatGPT API 翻訳を組み合わせ、
映像上に英語（上半分）と日本語訳（下半分）の二段組でオーバーレイ表示するサンプルコードです。

※事前に指定行数分のテキストを生成し、logフォルダに日付名のフォルダを作成して出力を保存した後、
　StyleGAN3 のリアルタイム生成に対して１行ずつオーバーレイ表示します。

※さらに、GPU性能の制約で約10fpsのStyleGAN生成フレームを、前後フレームの遷移（ブレンド）処理により
　約30fpsに見せるようにしています。なお、この遷移処理は config から --transition/--no-transition オプションで切り替え可能です。
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
import gc  # ガベージコレクションによるメモリリーク対策

# オプション：psutil がインストールされていればメモリ使用量ログ出力に利用
try:
    import psutil
    process = psutil.Process(os.getpid())
except ImportError:
    psutil = None

# API キーは config/api.py からインポート
from config.api import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# StyleGAN3 用の設定
from config.config import STYLEGAN_CONFIG

# latent 補間関数（latent_anima）およびその他ユーティリティ
from util.utilgan import latent_anima
# src/realtime_generate.py 内の必要関数（循環参照にならないよう注意）
from src.realtime_generate import infinite_latent_smooth, infinite_latent_random_walk, img_resize_for_cv2
# GPT 用モデル
from util.llm import GPT

# グローバル変数（事前生成テキスト用：main で保持するため、別スレッドは使わない）
text_lock = threading.Lock()

# ──────────────────────────────
# テキストサイズ計測用（PIL の getbbox を利用）
def get_text_size(font, text):
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)

# ──────────────────────────────
# 文字列を指定幅に合わせて改行する関数（単語単位でラップ、単語が長い場合は文字単位で分割）
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
                # 単語単体が max_width を超える場合、文字単位で分割
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

# ──────────────────────────────
# ChatGPT API を利用した翻訳関数
def translate_to_japanese(text):
    prompt = f"次の英語のテキストを、なるべく元の文体を保ったまま、神秘的な神話テキストとして日本語に翻訳してください:\n\n{text}"
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

# ──────────────────────────────
# 事前に指定行数分のテキストを生成し、改行処理を行った上でログに保存する関数
def pre_generate_text_lines(model_path, prompt, max_new_tokens, context_length, device, num_lines, log_dir, frame_width, font_scale, font_path):
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

    # フォント読み込み（font_scale に応じたフォントサイズ）
    font_size = int(32 * font_scale)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print("フォント読み込みエラー:", e)
        font = ImageFont.load_default()
    # 描画エリアの横幅の90%を最大文字幅とする
    max_text_width = int(frame_width * 0.9)

    generated_lines = []
    log_file_path = os.path.join(log_dir, "generated_texts.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for i in range(num_lines):
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

            # generated の型に応じてデコード
            if isinstance(generated, torch.Tensor):
                generated = generated.tolist()
            if isinstance(generated, (list, tuple)) and all(isinstance(token, int) for token in generated):
                en_text = tokenizer.decode(generated)
            else:
                en_text = str(generated)
            ja_text = translate_to_japanese(en_text)

            # 改行処理（wrap_text）を実施
            en_lines = wrap_text(en_text, font, max_text_width)
            ja_lines = wrap_text(ja_text, font, max_text_width)

            line_data = {"en": en_lines, "ja": ja_lines}
            generated_lines.append(line_data)

            # ログファイルへ出力（改行済みテキストは join して1行ずつ表示）
            log_file.write(f"Line {i+1}:\n")
            log_file.write("EN: " + " / ".join(en_lines) + "\n")
            log_file.write("JA: " + " / ".join(ja_lines) + "\n")
            log_file.write("\n")
            print(f"Pre-generated text line {i+1}/{num_lines}")
    return generated_lines

# ──────────────────────────────
# StyleGAN3 によるフレーム生成スレッド（従来と同様）
def stylegan_frame_generator(frame_queue, stop_event, config_args):
    device = torch.device(config_args["stylegan_gpu"])
    noise_seed = config_args["noise_seed"]
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    os.makedirs(config_args["out_dir"], exist_ok=True)

    # StyleGAN3 のネットワーク読み込み用パラメータ作成
    Gs_kwargs = dnnlib.EasyDict()
    for key in ["verbose", "size", "scale_type"]:
        Gs_kwargs[key] = config_args[key]

    # latmask の有無で latent blending 用パラメータを設定
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
        n_mult = 2
        nHW = [1,1]
        if osp.isfile(config_args["latmask"]):  # single file
            lmask = np.asarray([[img_read(config_args["latmask"])[:,:,0] / 255.]])
        elif osp.isdir(config_args["latmask"]):  # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(config_args["latmask"])]), 1)
        else:
            print(' !! Blending mask not found:', config_args["latmask"]); exit(1)
        if config_args["verbose"] is True:
            print(' Latent blending with mask', config_args["latmask"], lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1)
        lmask = torch.from_numpy(lmask).to(device)

    # Parse frames と fstep (例："200-25")
    frames_val, fstep_val = [int(x) for x in config_args["frames"].split('-')]

    # StyleGAN3 ネットワークの読み込み
    model_path = config_args["model"]
    pkl_name = os.path.splitext(model_path)[0]
    custom = False if '.pkl' in model_path.lower() else True
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        rot = True if ('-r-' in model_path.lower() or 'sg3r-' in model_path.lower()) else False
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim
    if c_dim > 0 and config_args["labels"] is not None:
        label = torch.zeros([1, c_dim], device=device)
        label_idx = min(int(config_args["labels"]), c_dim - 1)
        label[0, label_idx] = 1
    else:
        label = None

    if hasattr(Gs.synthesis, 'input'):
        if config_args["anim_trans"]:
            hw_centers = [np.linspace(-1 + 1/n, 1 - 1/n, n) for n in nHW]
            yy, xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in config_args["size"]]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * np.array(xscale) * 0.5 * config_args["shiftbase"]
            hw_scales = np.array([2. / n for n in nHW]) * config_args["shiftmax"]
            shifts = latent_anima((n_mult, 2), frames_val, fstep_val, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            shifts = np.zeros((1, n_mult, 2))
        if config_args["anim_rot"]:
            angles = latent_anima((n_mult, 1), frames_val, frames_val//4, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            angles = (angles - 0.5) * 180.
        else:
            angles = np.zeros((1, n_mult, 1))
        scale_array = np.array(config_args["affine_scale"], dtype=np.float32)
        scales = np.tile(scale_array, (shifts.shape[0], shifts.shape[1], 1))
        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        scales = torch.from_numpy(scales).to(device)
        trans_params = list(zip(shifts, angles, scales))
    else:
        trans_params = None

    if hasattr(Gs.synthesis, 'input'):
        first_layer_channels = Gs.synthesis.input.channels
        first_layer_size = Gs.synthesis.input.size
        if isinstance(first_layer_size, (list, tuple, np.ndarray)):
            h, w = first_layer_size[0], first_layer_size[1]
        else:
            h, w = first_layer_size, first_layer_size
        shape_for_dconst = [1, first_layer_channels, h, w]
        if config_args["digress"] != 0:
            dconst_list = []
            for i in range(n_mult):
                dc_tmp = config_args["digress"] * latent_anima(shape_for_dconst, frames_val, fstep_val,
                                                                 cubic=True, seed=noise_seed, verbose=False)
                dconst_list.append(dc_tmp)
            dconst = np.concatenate(dconst_list, axis=1)
        else:
            dconst = np.zeros([shifts.shape[0], 1, first_layer_channels, h, w])
        dconst = torch.from_numpy(dconst).to(device).to(torch.float32)
    else:
        dconst = None

    with torch.no_grad():
        if custom and hasattr(Gs.synthesis, 'input'):
            dummy_trans_param = (torch.zeros([1,2], device=device),
                                 torch.zeros([1,1], device=device),
                                 torch.ones([1,2], device=device))
            _ = Gs(torch.randn([1, z_dim], device=device), label, lmask[0] if lmask is not None else None,
                   dummy_trans_param, dconst[0] if dconst is not None else None, noise_mode='const')
        else:
            _ = Gs(torch.randn([1, z_dim], device=device), label,
                   truncation_psi=config_args["trunc"], noise_mode='const')

    frame_idx_local = 0
    frame_idx = 0
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
                latmask = lmask[frame_idx_local % len(lmask)] if lmask is not None else None
                dconst_current = dconst[frame_idx % dconst.shape[0]] if dconst is not None else None
                trans_param = trans_params[frame_idx % len(trans_params)] if trans_params is not None else None
                out = Gs(z_current, label, latmask, trans_param, dconst_current,
                         truncation_psi=config_args["trunc"], noise_mode='const')
            else:
                out = Gs(z_current, label, None, truncation_psi=config_args["trunc"], noise_mode='const')
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0,255).to(torch.uint8)
        out_np = out[0].cpu().numpy()[..., ::-1]  # BGR

        # キューが満杯の場合は古いフレームを破棄して新フレームを入れる
        try:
            frame_queue.put(out_np, block=False)
        except queue.Full:
            try:
                frame_queue.get_nowait()  # 先頭の古いフレームを破棄
            except queue.Empty:
                pass
            frame_queue.put(out_np, block=False)

        frame_idx_local += 1
        frame_idx += 1

# ──────────────────────────────
# 中間フレームを生成する関数（軽量なベクトル演算によるブレンド）
def generate_transition_frame(frame_a, frame_b, t):
    """
    frame_a, frame_b: uint8 の numpy 配列（H×W×3）
    t: 0～1 の補間パラメータ（t=0 で frame_a、t=1 で frame_b）
    """
    fa = frame_a.astype(np.float32)
    fb = frame_b.astype(np.float32)
    linear_blend = (1 - t) * fa + t * fb
    multiply_blend = (fa * fb) / 255.0
    brightness = np.mean(fb, axis=2, keepdims=True) / 255.0
    whiteness = np.clip((brightness - 0.6) / 0.4, 0, 1)
    noise = np.random.uniform(0, 1, size=whiteness.shape)
    base_alpha = t * (1 - t) * 4
    alpha = base_alpha * (0.5 + 0.8 * whiteness * (0.5 + 0.5 * noise))
    alpha = np.clip(alpha, 0, 1)
    result = (1 - alpha) * linear_blend + alpha * multiply_blend
    return np.clip(result, 0, 255).astype(np.uint8)

# ──────────────────────────────
# 【新規】 改行済みテキスト行リストを用いてオーバーレイ画像を生成する関数
def create_text_overlay(frame_shape, texts, font_scale, thickness, font_path):
    """
    frame_shape: (H, W, ...) のタプル
    texts: {"en": [行1, 行2, ...], "ja": [行1, 行2, ...]} の辞書（各言語とも既に改行済み）
    font_scale: フォントサイズのスケール
    thickness: テキストの縁取りの太さ
    font_path: フォントファイルのパス
    """
    h, w = frame_shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(font_path, int(32 * font_scale))
    except Exception as e:
        print("フォント読み込みエラー:", e)
        font = ImageFont.load_default()

    en_lines = texts.get("en", [])
    ja_lines = texts.get("ja", [])
    gap = 5

    # 英文ブロックの高さ計算
    en_block_h = sum(get_text_size(font, line)[1] for line in en_lines) + (len(en_lines)-1)*gap if en_lines else 0
    # 日本語ブロックの高さ計算
    ja_block_h = sum(get_text_size(font, line)[1] for line in ja_lines) + (len(ja_lines)-1)*gap if ja_lines else 0

    top_region_center = h // 4
    bottom_region_center = h * 3 // 4
    en_y0 = top_region_center - en_block_h // 2
    ja_y0 = bottom_region_center - ja_block_h // 2

    # 英文を中央寄せで描画
    y = en_y0
    for line in en_lines:
        line_w, line_h = get_text_size(font, line)
        x = (w - line_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                  stroke_width=thickness, stroke_fill="black")
        y += line_h + gap

    # 日本語を中央寄せで描画
    y = ja_y0
    for line in ja_lines:
        line_w, line_h = get_text_size(font, line)
        x = (w - line_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                  stroke_width=thickness, stroke_fill="black")
        y += line_h + gap

    overlay_np = np.array(overlay)
    return overlay_np

# ──────────────────────────────
# 【新規】 オーバーレイ画像とフレームをブレンドする関数
def blend_overlay(frame, overlay):
    """
    frame: BGR (uint8) の NumPy 配列
    overlay: RGBA (uint8) のオーバーレイ画像
    """
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA).astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    alpha = overlay_float[:, :, 3:4] / 255.0
    blended = frame_bgra.copy()
    blended[:, :, :3] = frame_bgra[:, :, :3] * (1 - alpha) + overlay_float[:, :, :3] * alpha
    blended_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    return blended_bgr

# ──────────────────────────────
# Click オプション（各種パラメータは STYLEGAN_CONFIG をデフォルトに）
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
@click.option('--sg_gpu', type=str, default=STYLEGAN_CONFIG['sg_gpu'], help="GPU for StyleGAN3")
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
@click.option('--gpt-model', type=str, default=STYLEGAN_CONFIG['gpt_model'], help="GPT model checkpoint path")
@click.option('--gpt-prompt', type=str, default=STYLEGAN_CONFIG['gpt_prompt'], help="GPT generation prompt")
@click.option('--max-new-tokens', type=int, default=STYLEGAN_CONFIG['max_new_tokens'], help="maximum new tokens for GPT")
@click.option('--context-length', type=int, default=STYLEGAN_CONFIG['context_length'], help="GPT context length")
@click.option('--gpt-gpu', type=str, default=STYLEGAN_CONFIG['gpt_gpu'], help="GPU for GPT")
@click.option('--display-time', type=float, default=STYLEGAN_CONFIG['display_time'], help="display time for generated text (seconds)")
@click.option('--clear-time', type=float, default=STYLEGAN_CONFIG['clear_time'], help="clear time for text (seconds)")
# オーバーレイテキスト用フォントパラメータ
@click.option('--font-scale', type=float, default=STYLEGAN_CONFIG['default_font_scale'], help="default font scale for overlay text")
@click.option('--font-thickness', type=int, default=STYLEGAN_CONFIG['default_font_thickness'], help="default font thickness for overlay text")
# 事前生成するテキスト行数
@click.option('--text-lines', type=int, default=10, help="Number of text lines to pre-generate")
# フレーム間トランジション処理のオン/オフ（デフォルトはオン）
@click.option('--transition/--no-transition', default=True, help="Enable transition interpolation for smoother frame rate (simulate 30fps)")
def cli(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc,
        save_lat, verbose, noise_seed, frames, cubic, gauss, anim_trans, anim_rot, shiftbase,
        shiftmax, digress, affine_scale, framerate, prores, variations, method,
        gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_gpu, display_time, clear_time,
        sg_gpu, font_scale, font_thickness, text_lines, transition):
    """
    StyleGAN3 によるリアルタイム映像生成と、事前に GPT により生成・翻訳したテキストを組み合わせ、
    映像上に英語（上半分）と日本語訳（下半分）でオーバーレイ表示します。

    起動時に指定行数分のテキストを生成し、log フォルダに日付名のフォルダを作成して出力を保存します。
    また、GPU の制約により StyleGAN の生成フレームは約10fpsですが、前後フレームのブレンドにより
    中間フレームを生成して約30fpsに見せるアニメーション遷移処理を、--transition/--no-transition で指定できます。

    さらに、事前生成時にテキストの描画領域からはみ出るかを判定して改行済みテキストを保存することで、
    オーバーレイ時の重い改行計算を省いてリアルタイム性を追求します。
    """
    try:
        if "-" in size:
            w, h = size.split("-")
        elif "x" in size:
            w, h = size.split("x")
        elif "X" in size:
            w, h = size.split("X")
        elif "," in size:
            w, h = size.split(",")
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
        "stylegan_gpu": sg_gpu,
        "font_scale": font_scale,
        "font_thickness": font_thickness
    }

    # 事前に GPT でテキストを生成し、ログ出力フォルダを作成する
    log_base = "log"
    os.makedirs(log_base, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_base, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    gpt_device = torch.device(gpt_gpu if torch.cuda.is_available() else "cpu")
    print("事前にテキストを生成中...")
    # 画面の横幅は size_parsed[1]、font_scale と font_path は各パラメータから取得
    pre_generated_texts = pre_generate_text_lines(
        gpt_model, gpt_prompt, max_new_tokens, context_length,
        gpt_device, text_lines, log_dir,
        frame_width=size_parsed[1],
        font_scale=font_scale,
        font_path=STYLEGAN_CONFIG['font_path']
    )
    print("テキストの事前生成完了。")

    # StyleGAN3 用フレーム生成スレッド開始
    frame_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()
    gan_thread = threading.Thread(target=stylegan_frame_generator,
                                  args=(frame_queue, stop_event, config_args),
                                  daemon=True)
    gan_thread.start()

    print("リアルタイムプレビュー開始（'q' キーで終了）")
    window_name = "StyleGAN3 + GPT Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 定期実行用の変数設定
    last_context_reinit_time = time.time()
    context_reinit_interval = 300
    last_gc_time = time.time()
    gc_interval = 60

    fps_count = 0
    t0 = time.time()
    curr_frame = frame_queue.get()  # blocking
    prev_frame = curr_frame.copy()
    last_frame_update = time.time()
    display_interval_ms = 33  # 約30fps
    stylegan_interval = 0.1     # StyleGAN3 のフレーム間隔（約10fps）

    # テキスト表示用の変数
    current_text_idx = 0
    text_visible = True
    last_text_change = time.time()
    # オーバーレイ画像キャッシュ用変数
    current_overlay = None
    current_text = None

    while True:
        now = time.time()

        if now - last_gc_time >= gc_interval:
            gc.collect()
            last_gc_time = now
            if psutil is not None:
                mem = process.memory_info().rss / (1024 * 1024)
                print(f"\n[GC] メモリ使用量: {mem:.2f} MB")

        if now - last_context_reinit_time >= context_reinit_interval:
            cv2.destroyAllWindows()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            last_context_reinit_time = now
            print("\n[Context Reinit] OpenCV ウィンドウコンテキストを再初期化しました。")

        try:
            while True:
                new_frame = frame_queue.get_nowait()
                prev_frame = curr_frame.copy()
                curr_frame = new_frame
                last_frame_update = time.time()
        except queue.Empty:
            pass

        t = (time.time() - last_frame_update) / stylegan_interval
        if t > 1:
            t = 1.0

        if transition:
            disp_frame = generate_transition_frame(prev_frame, curr_frame, t)
        else:
            disp_frame = curr_frame

        # テキスト表示のオン／オフ（display_time, clear_time に基づく）
        if text_visible and now - last_text_change >= display_time:
            text_visible = False
            last_text_change = now
        elif not text_visible and now - last_text_change >= clear_time:
            current_text_idx = (current_text_idx + 1) % len(pre_generated_texts)
            text_visible = True
            last_text_change = now

        # 表示するテキスト：テキスト非表示の場合は空のリストを設定
        if text_visible:
            new_text = pre_generated_texts[current_text_idx]
        else:
            new_text = {"en": [], "ja": []}

        # テキスト内容が変わった場合のみオーバーレイ画像を再生成
        if new_text != current_text:
            current_text = new_text
            current_overlay = create_text_overlay(disp_frame.shape, current_text, font_scale, font_thickness, STYLEGAN_CONFIG['font_path'])

        if current_overlay is not None:
            frame_with_text = blend_overlay(disp_frame.copy(), current_overlay)
        else:
            frame_with_text = disp_frame.copy()

        frame_with_text = img_resize_for_cv2(frame_with_text)
        cv2.imshow(window_name, frame_with_text)

        fps_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"\r{fps_count / elapsed:.2f} fps", end="")
            sys.stdout.flush()
            t0 = time.time()
            fps_count = 0

        key = cv2.waitKey(display_interval_ms) & 0xFF
        if key == ord('q'):
            print("\n終了します。")
            stop_event.set()
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    cli()
