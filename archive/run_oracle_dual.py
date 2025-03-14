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
import os.path as osp
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
from PIL import Image, ImageDraw, ImageFont

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

# グローバル変数（表示テキスト：英語原文と日本語訳）
current_text = {"en": "", "ja": ""}
text_lock = threading.Lock()


def get_text_size(font, text):
    # font.getbbox(text) returns (x0, y0, x1, y1)
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)

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
# 単語単位で自動改行するヘルパー関数
def wrap_text(text, max_width, font, font_scale, thickness):
    """
    与えられたテキストを、get_text_size() を用いて max_width (ピクセル) を超えないように
    単語単位で改行し、行のリストを返します。
    単一の単語（または日本語などスペースがない場合）は、全体が max_width に収まればそのまま、
    収まらなければできるだけ多くの文字を１行に詰め、余った部分は次行にするようにします。
    """
    # もしテキスト内にスペースがなければ（＝日本語などの場合）
    if " " not in text:
        if get_text_size(font, text)[0] <= max_width:
            return [text]
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            if get_text_size(font, test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return lines

    # テキスト内にスペースがある場合（＝英語など）
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
                # 単一の単語が max_width を超える場合は文字単位で改行
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

    # ここでは、StyleGAN3 のネットワーク読み込みに必要なキーのみを抜粋して Gs_kwargs を作成
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
        if osp.isfile(config_args["latmask"]): # single file
            lmask = np.asarray([[img_read(config_args["latmask"])[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(config_args["latmask"]): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(config_args["latmask"])]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', config_args["latmask"]); exit(1)
        if a.verbose is True: print(' Latent blending with mask', config_args["latmask"], lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [n,2,h,w]
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

    # NEW SG3 の場合、追加の変換パラメータを生成
    if hasattr(Gs.synthesis, 'input'):
        # 平行移動
        if config_args["anim_trans"]:
            hw_centers = [np.linspace(-1 + 1/n, 1 - 1/n, n) for n in nHW]
            yy, xx = np.meshgrid(*hw_centers)
            # Gs.img_resolution は画像解像度（正方形の場合）
            xscale = [s / Gs.img_resolution for s in config_args["size"]]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * np.array(xscale) * 0.5 * config_args["shiftbase"]
            hw_scales = np.array([2. / n for n in nHW]) * config_args["shiftmax"]
            shifts = latent_anima((n_mult, 2), frames_val, fstep_val, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            shifts = np.zeros((1, n_mult, 2))
        # 回転
        if config_args["anim_rot"]:
            angles = latent_anima((n_mult, 1), frames_val, frames_val//4, uniform=True,
                                  cubic=config_args["cubic"], gauss=config_args["gauss"],
                                  seed=noise_seed, verbose=False)
            angles = (angles - 0.5) * 180.
        else:
            angles = np.zeros((1, n_mult, 1))
        # 拡大率（affine_scale は [scale_y, scale_x] としてパース済み）
        scale_array = np.array(config_args["affine_scale"], dtype=np.float32)  # (scale_y, scale_x)
        scales = np.tile(scale_array, (shifts.shape[0], shifts.shape[1], 1))
        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        scales = torch.from_numpy(scales).to(device)
        trans_params = list(zip(shifts, angles, scales))
    else:
        trans_params = None

    # 歪み（digress）の計算
    if hasattr(Gs.synthesis, 'input'):
        first_layer_channels = Gs.synthesis.input.channels
        first_layer_size = Gs.synthesis.input.size
        if isinstance(first_layer_size, (list, tuple, np.ndarray)):
            h, w = first_layer_size[0], first_layer_size[1]
        else:
            h, w = first_layer_size, first_layer_size
        shape_for_dconst = [1, first_layer_channels, h, w]
        # print("debug shape_for_dconst =", shape_for_dconst)
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

    # 初回ウォームアップ推論
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

    # メインループ
    frame_idx_local = 0
    frame_idx = 0
    # latent 生成モードの選択
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
                # 対応する dconst, trans_param を選択
                dconst_current = dconst[frame_idx % dconst.shape[0]] if dconst is not None else None
                trans_param = trans_params[frame_idx % len(trans_params)] if trans_params is not None else None
                out = Gs(z_current, label, latmask, trans_param, dconst_current,
                         truncation_psi=config_args["trunc"], noise_mode='const')
            else:
                out = Gs(z_current, label, None, truncation_psi=config_args["trunc"], noise_mode='const')
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0,255).to(torch.uint8)
        out_np = out[0].cpu().numpy()[..., ::-1]  # BGR
        frame_queue.put(out_np)
        frame_idx_local += 1
        frame_idx += 1


# ──────────────────────────────
# --- 変更後の overlay_text_on_frame 関数 ---
def overlay_text_on_frame(frame, texts, font_scale, thickness, font_path=STYLEGAN_CONFIG['font_path'], color=STYLEGAN_CONFIG['font_color']):
    # BGR -> RGB に変換して PIL Image にする
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    frame_w, frame_h = image.size
    max_text_width = int(frame_w * 0.9)

    # PIL 用の自動改行関数
    def wrap_text_pil(text, font):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if get_text_size(font, test_line)[0] <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # 単一の単語が長い場合は文字単位で改行
                    sub_line = ""
                    for char in word:
                        test_sub_line = sub_line + char
                        if get_text_size(font, test_line)[0] <= max_text_width:
                            sub_line = test_sub_line
                        else:
                            lines.append(sub_line)
                            sub_line = char
                    current_line = sub_line
        if current_line:
            lines.append(current_line)
        return lines

    # font_scale をもとにフォントサイズを設定（例: 基準 32）
    font_size = int(32 * font_scale)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print("フォント読み込みエラー:", e)
        font = ImageFont.load_default()

    en_text = texts.get("en", "")
    ja_text = texts.get("ja", "")
    en_lines = wrap_text_pil(en_text, font)
    ja_lines = wrap_text_pil(ja_text, font)

    gap = 5
    def get_block_height(lines, font):
        h_total = 0
        for line in lines:
            h_total += get_text_size(font, line)[1] + gap
        return h_total - gap if lines else 0

    en_block_h = get_block_height(en_lines, font)
    ja_block_h = get_block_height(ja_lines, font)
    top_region_center = frame_h // 4
    bottom_region_center = frame_h * 3 // 4
    en_y0 = top_region_center - en_block_h // 2
    ja_y0 = bottom_region_center - ja_block_h // 2

    def draw_lines(lines, y0, font, fill, stroke_fill="black", stroke_width=2):
        y = y0
        for line in lines:
            line_w, line_h = get_text_size(font, line)
            x = (frame_w - line_w) // 2
            draw.text((x, y), line, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw.text((x, y), line, font=font, fill=fill)
            y += line_h + gap

    draw_lines(en_lines, en_y0, font, fill=color)
    draw_lines(ja_lines, ja_y0, font, fill=color)

    # PIL Image -> OpenCV (BGR) に戻す
    result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return result


# ──────────────────────────────
# Click オプション（StyleGAN3 用パラメータは STYLEGAN_CONFIG をデフォルトに）
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
@click.option('--gpt-model', type=str, default=STYLEGAN_CONFIG['gpt_model'], help="GPT model checkpoint path")
@click.option('--gpt-prompt', type=str, default=STYLEGAN_CONFIG['gpt_prompt'], help="GPT generation prompt")
@click.option('--max-new-tokens', type=int, default=STYLEGAN_CONFIG['max_new_tokens'], help="maximum new tokens for GPT")
@click.option('--context-length', type=int, default=STYLEGAN_CONFIG['context_length'], help="GPT context length")
@click.option('--gpt-gpu', type=str, default=STYLEGAN_CONFIG['gpt_gpu'], help="GPU for GPT")
@click.option('--display-time', type=float, default=STYLEGAN_CONFIG['display_time'], help="display time for generated text (seconds)")
@click.option('--clear-time', type=float, default=STYLEGAN_CONFIG['clear_time'], help="clear time for text (seconds)")

@click.option('--font-scale', type=float, default=STYLEGAN_CONFIG['default_font_scale'], help="default font scale for overlay text")
@click.option('--font-thickness', type=int, default=STYLEGAN_CONFIG['default_font_thickness'], help="default font thickness for overlay text")


def cli(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc,
        save_lat, verbose, noise_seed, frames, cubic, gauss, anim_trans, anim_rot, shiftbase,
        shiftmax, digress, affine_scale, framerate, prores, variations, method,
        gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_gpu, display_time, clear_time, font_scale, font_thickness):
    """
    StyleGAN3 によるリアルタイム映像生成と GPT によるテキスト生成＋ChatGPT API 翻訳を組み合わせ、
    映像上に英語（上半分）と日本語訳（下半分）でオーバーレイ表示します。
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
        "stylegan_gpu": "cuda:0",
        "font_scale": font_scale,
        "font_thickness": font_thickness
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
        frame_with_text = overlay_text_on_frame(frame.copy(), texts, font_scale, font_thickness)
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
