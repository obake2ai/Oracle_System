#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
このコードは、StyleGAN3 によるリアルタイム映像生成と GPT によるテキスト生成＋翻訳を組み合わせ、
映像上に日本語（上部、半透明）と英語（下部、半透明）のテキストをオーバーレイ表示します。
また、--debug オプションを指定すると、cProfile によるボトルネックのプロファイリングを実施し、
終了時に累積時間が一定以上（ここでは 0.1 秒以上）の関数のみを表示します。
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

# psutil（メモリ監視用）
try:
    import psutil
    process = psutil.Process(os.getpid())
except ImportError:
    psutil = None

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

from config.api import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

from config.config import STYLEGAN_CONFIG
from config.prompts import CHATGPT_PROMPTS
from util.utilgan import latent_anima
from src.realtime_generate import infinite_latent_smooth, infinite_latent_random_walk, img_resize_for_cv2
from util.llm import GPT

# グローバル変数（事前生成テキスト用）
text_lock = threading.Lock()

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
    # ※キー "translate" を利用してプロンプトを取得（必要に応じて修正してください）
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

def pre_generate_text_lines(model_path, prompt, max_new_tokens, context_length, device, num_lines, log_dir, frame_width, font_scale, font_path_en, font_path_ja):
    tokenizer = tiktoken.get_encoding('gpt2')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model_state_dict'].items()}
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
    generated_lines = []
    log_file_path = os.path.join(log_dir, "generated_texts.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for i in range(num_lines):
            input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]
            if isinstance(generated, torch.Tensor):
                generated = generated.tolist()
            if isinstance(generated, (list, tuple)) and all(isinstance(token, int) for token in generated):
                en_text = tokenizer.decode(generated)
            else:
                en_text = str(generated)
            ja_text = translate_to_japanese(en_text)
            en_lines = wrap_text(en_text, font_en, max_text_width)
            ja_lines = wrap_text(ja_text, font_ja, max_text_width)
            line_data = {"en": en_lines, "ja": ja_lines}
            generated_lines.append(line_data)
            log_file.write(f"Line {i+1}:\n")
            log_file.write("EN: " + " / ".join(en_lines) + "\n")
            log_file.write("JA: " + " / ".join(ja_lines) + "\n\n")
            print(f"Pre-generated text line {i+1}/{num_lines}")
    return generated_lines

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
# StyleGAN3 フレーム生成スレッド
# -------------------------------
def stylegan_frame_generator(frame_queue, stop_event, config_args):
    device = torch.device(config_args["stylegan_gpu"])
    noise_seed = config_args["noise_seed"]
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)
    os.makedirs(config_args["out_dir"], exist_ok=True)
    Gs_kwargs = dnnlib.EasyDict()
    for key in ["verbose", "size", "scale_type"]:
        Gs_kwargs[key] = config_args[key]
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
        nHW = [1, 1]
        if osp.isfile(config_args["latmask"]):
            lmask = np.asarray([[img_read(config_args["latmask"])[:,:,0] / 255.]])
        elif osp.isdir(config_args["latmask"]):
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(config_args["latmask"])]), 1)
        else:
            print(' !! Blending mask not found:', config_args["latmask"]); exit(1)
        if config_args["verbose"]:
            print(' Latent blending with mask', config_args["latmask"], lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1)
        lmask = torch.from_numpy(lmask).to(device)
    frames_val, fstep_val = [int(x) for x in config_args["frames"].split('-')]
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
        print("random")
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=noise_seed, step_size=0.02)
    else:
        print("smooth")
        latent_gen = infinite_latent_smooth(z_dim=z_dim, device=device,
                                            cubic=config_args["cubic"],
                                            gauss=config_args["gauss"],
                                            seed=noise_seed,
                                            chunk_size=config_args["chunk_size"],
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
        out_np = out[0].cpu().numpy()[..., ::-1]
        try:
            frame_queue.put(out_np, block=False)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(out_np, block=False)
        frame_idx_local += 1
        frame_idx += 1

# -------------------------------
# create_text_overlay: テキストオーバーレイ画像生成（透過度付き）
# -------------------------------
def create_text_overlay(frame_shape, texts, font_scale, thickness, font_path_en, font_path_ja):
    h, w = frame_shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    try:
        font_en = ImageFont.truetype(font_path_en, int(32 * font_scale))
    except Exception as e:
        print("英語フォント読み込みエラー:", e)
        font_en = ImageFont.load_default()
    try:
        font_ja = ImageFont.truetype(font_path_ja, int(32 * font_scale))
    except Exception as e:
        print("日本語フォント読み込みエラー:", e)
        font_ja = ImageFont.load_default()
    ja_lines = texts.get("ja", [])
    en_lines = texts.get("en", [])
    default_line_height = get_text_size(font_ja, "A")[1]
    ja_block_height = sum(get_text_size(font_ja, line)[1] for line in ja_lines) if ja_lines else 0
    en_block_height = sum(get_text_size(font_en, line)[1] for line in en_lines) if en_lines else 0
    combined_height = ja_block_height + (default_line_height if ja_lines and en_lines else 0) + en_block_height
    start_y = (h - combined_height) // 2
    y = start_y
    # 透過度 90% (alpha=230)
    text_fill = (255, 255, 255, 230)
    stroke_fill = (0, 0, 0, 230)
    for line in ja_lines:
        line_w, line_h = get_text_size(font_ja, line)
        x = (w - line_w) // 2
        draw = ImageDraw.Draw(overlay)
        draw.text((x, y), line, font=font_ja, fill=text_fill,
                  stroke_width=thickness, stroke_fill=stroke_fill)
        y += line_h
    if ja_lines and en_lines:
        y += default_line_height
    for line in en_lines:
        line_w, line_h = get_text_size(font_en, line)
        x = (w - line_w) // 2
        draw = ImageDraw.Draw(overlay)
        draw.text((x, y), line, font=font_en, fill=text_fill,
                  stroke_width=thickness, stroke_fill=stroke_fill)
        y += line_h
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
# CLI 部分（各種オプション含む）
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
@click.option('--sg_gpu', type=str, default=STYLEGAN_CONFIG['sg_gpu'], help="GPU for StyleGAN3")
@click.option('--frames', type=str, default=STYLEGAN_CONFIG['frames'], help="total frames and interpolation step (e.g., 200-25)")
@click.option('--cubic', is_flag=True, default=STYLEGAN_CONFIG['cubic'], help="use cubic splines for smoothing")
@click.option('--gauss', is_flag=True, default=STYLEGAN_CONFIG['gauss'], help="use Gaussian smoothing")
@click.option('--anim_trans', is_flag=True, default=STYLEGAN_CONFIG['anim_trans'], help="add translation animation")
@click.option('--anim_rot', is_flag=True, default=STYLEGAN_CONFIG['anim_rot'], help="add rotation animation")
@click.option('--shiftbase', type=float, default=STYLEGAN_CONFIG['shiftbase'], help="shift to the tile center")
@click.option('--shiftmax', type=float, default=STYLEGAN_CONFIG['shiftmax'], help="random walk around tile center")
@click.option('--digress', type=float, default=STYLEGAN_CONFIG['digress'], help="distortion strength")
@click.option('--affine_scale', type=str, default=STYLEGAN_CONFIG['affine_scale'], help="affine scale (e.g., 1.0-1.0)")
@click.option('--framerate', type=int, default=STYLEGAN_CONFIG['framerate'], help="frame rate")
@click.option('--prores', is_flag=True, default=STYLEGAN_CONFIG['prores'], help="output video in ProRes format")
@click.option('--variations', type=int, default=STYLEGAN_CONFIG['variations'], help="number of variations")
@click.option('--method', type=click.Choice(["smooth", "random_walk"]), default=STYLEGAN_CONFIG['method'], help="infinite realtime generation method")
@click.option('--chunk_size', type=int, default=STYLEGAN_CONFIG['chunk_size'], help="step size for infinite realtime generation method")
@click.option('--gpt-model', type=str, default=STYLEGAN_CONFIG['gpt_model'], help="GPT model checkpoint path")
@click.option('--gpt-prompt', type=str, default=STYLEGAN_CONFIG['gpt_prompt'], help="GPT generation prompt")
@click.option('--max-new-tokens', type=int, default=STYLEGAN_CONFIG['max_new_tokens'], help="maximum new tokens for GPT")
@click.option('--context-length', type=int, default=STYLEGAN_CONFIG['context_length'], help="GPT context length")
@click.option('--gpt-gpu', type=str, default=STYLEGAN_CONFIG['gpt_gpu'], help="GPU for GPT")
@click.option('--display-time', type=float, default=STYLEGAN_CONFIG['display_time'], help="display time for generated text (seconds)")
@click.option('--clear-time', type=float, default=STYLEGAN_CONFIG['clear_time'], help="clear time for text (seconds)")
@click.option('--font-scale', type=float, default=STYLEGAN_CONFIG['default_font_scale'], help="default font scale for overlay text")
@click.option('--font-thickness', type=int, default=STYLEGAN_CONFIG['default_font_thickness'], help="default font thickness for overlay text")
@click.option('--text-lines', type=int, default=10, help="Number of text lines to pre-generate")
@click.option('--debug', is_flag=True, default=False, help="Enable profiling of bottlenecks")
@click.option('--fullscreen/--windowed', default=True, help="Use fullscreen mode if enabled; otherwise use window mode")
def cli(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc,
        save_lat, verbose, noise_seed, frames, cubic, gauss, anim_trans, anim_rot, shiftbase,
        shiftmax, digress, affine_scale, framerate, prores, variations, method, chunk_size,
        gpt_model, gpt_prompt, max_new_tokens, context_length, gpt_gpu, display_time, clear_time,
        sg_gpu, font_scale, font_thickness, text_lines, debug, fullscreen):
    # プロファイリング開始（--debug オプションが有効な場合）
    profiler = None
    if debug:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
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
            "chunk_size": chunk_size,
            "stylegan_gpu": sg_gpu,
            "font_scale": font_scale,
            "font_thickness": font_thickness
        }
        log_base = "log"
        os.makedirs(log_base, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_base, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        gpt_device = torch.device(gpt_gpu if torch.cuda.is_available() else "cpu")

        # -------------------------
        # 事前テキスト生成（最大60秒待機）
        # -------------------------
        print("事前にテキストを生成中...")
        pre_generated_texts = []
        text_thread_result = {}
        def generate_text():
            result = pre_generate_text_lines(
                gpt_model, gpt_prompt, max_new_tokens, context_length,
                gpt_device, text_lines, log_dir,
                frame_width=size_parsed[1],
                font_scale=font_scale,
                font_path_en=STYLEGAN_CONFIG['font_path_en'],
                font_path_ja=STYLEGAN_CONFIG['font_path_ja']
            )
            text_thread_result['result'] = result
        text_thread = threading.Thread(target=generate_text, daemon=True)
        text_thread.start()
        text_thread.join(timeout=60)
        if text_thread.is_alive():
            print("テキスト生成タイムアウト。60秒以内に更新がなかったため、テキストなしで開始します。")
            pre_generated_texts = [{"en": [], "ja": []}]
        else:
            pre_generated_texts = text_thread_result.get('result', [{"en": [], "ja": []}])
        print("テキストの事前生成完了。")

        frame_queue = queue.Queue(maxsize=30)
        stop_event = threading.Event()
        gan_thread = threading.Thread(target=stylegan_frame_generator,
                                      args=(frame_queue, stop_event, config_args),
                                      daemon=True)
        gan_thread.start()
        print("リアルタイムプレビュー開始（'q' キーで終了）")
        screen_width = actual_screen_width
        screen_height = actual_screen_height
        window_name = "Display"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # リフレッシュ間隔を10分（600秒）に設定
        context_reinit_interval = 600
        last_context_reinit_time = time.time()
        last_gc_time = time.time()
        gc_interval = 60
        fps_count = 0
        t0 = time.time()
        curr_frame = frame_queue.get()  # blocking
        prev_frame = curr_frame.copy()
        last_frame_update = time.time()
        display_interval_ms = 33
        stylegan_interval = 0.1
        current_text_idx = 0
        text_visible = True
        last_text_change = time.time()
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
            # ウィンドウの再生成は行わず、プロパティの更新のみでリフレッシュする
            if now - last_context_reinit_time >= context_reinit_interval:
                print("\n[Context Reinit] OpenCV ウィンドウのプロパティを更新しました。")
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                last_context_reinit_time = now
            try:
                while True:
                    new_frame = frame_queue.get_nowait()
                    prev_frame = curr_frame  # 必要なら前のフレームは保持
                    curr_frame = new_frame
                    last_frame_update = time.time()
                # キューが空なら抜ける
            except queue.Empty:
                pass

            t = (time.time() - last_frame_update) / stylegan_interval
            if t > 1:
                t = 1.0

            # テキスト表示のON/OFFのタイミングはそのまま
            if text_visible and now - last_text_change >= display_time:
                text_visible = False
                last_text_change = now
            elif not text_visible and now - last_text_change >= clear_time:
                current_text_idx = (current_text_idx + 1) % len(pre_generated_texts)
                text_visible = True
                last_text_change = now

            # 表示すべきテキストを決定
            if text_visible:
                new_text = pre_generated_texts[current_text_idx]
            else:
                new_text = {"en": [], "ja": []}

            # 既存のキャッシュ（current_overlay）が存在し、かつフレームサイズが変わっていなければ再生成しない
            if new_text != current_text or current_overlay is None or current_overlay.shape[:2] != curr_frame.shape[:2]:
                current_text = new_text
                current_overlay = create_text_overlay(curr_frame.shape, current_text, font_scale, font_thickness,
                                                      STYLEGAN_CONFIG['font_path_en'], STYLEGAN_CONFIG['font_path_ja'])

            # blend_overlay 内で変換・コピーを実施しているため、ここではそのまま curr_frame を渡す
            if current_overlay is not None:
                frame_with_text = blend_overlay(curr_frame, current_overlay)
            else:
                frame_with_text = curr_frame

            # その後、letterbox_frame 等の処理はそのまま
            letterboxed_frame = letterbox_frame(frame_with_text, screen_width, screen_height)
            cv2.imshow(window_name, letterboxed_frame)

            # fps計測やキー入力処理はそのまま
            fps_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                print(f"\r{fps_count / elapsed:.2f} fps", end="")
                t0 = time.time()
                fps_count = 0
            key = cv2.waitKey(display_interval_ms) & 0xFF
            if key == ord('q'):
                print("\n終了します。")
                stop_event.set()
                break

        cv2.destroyAllWindows()  # ループ終了時のみウィンドウを破棄
    finally:
        # プロファイリング終了時、累積時間が閾値以上の関数のみを表示する
        if profiler is not None:
            profiler.disable()
            ps = pstats.Stats(profiler)
            ps.strip_dirs().sort_stats("cumtime")
            threshold = 0.1  # 0.1秒以上かかった関数のみ表示（必要に応じて調整）
            print("\n=== プロファイリング結果 (累積時間が {:.3f} 秒以上の関数のみ) ===".format(threshold))
            sorted_stats = sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True)
            for func, stat in sorted_stats:
                cc, nc, tt, ct, callers = stat
                if ct >= threshold:
                    func_desc = f"{func[0]}:{func[1]}({func[2]})"
                    print(f"{func_desc:60s}  Call Count: {nc:4d}  Total Time: {tt:8.6f}  Cumulative Time: {ct:8.6f}")

if __name__ == '__main__':
    cli()
