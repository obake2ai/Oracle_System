import os
import os.path as osp
import random
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy
from torch_utils import misc

from util.utilgan import latent_anima, basename, img_read, img_list
from util.progress_bar import progbar

# Colab or local preview
import cv2
import time
import sys
from IPython.display import display, clear_output
from PIL import Image

import queue
import threading

import click
from types import SimpleNamespace

torch.backends.cudnn.benchmark = True

###############################################################################
# クリックによるコマンドライン引数定義
###############################################################################

@click.command()
@click.option("-o", "--out_dir", default="_out", show_default=True,
              help="出力先ディレクトリ")
@click.option("-m", "--model", default="models/embryo-stylegan3-r-network-snapshot-000096.pkl", show_default=True,
              help="pkl チェックポイントファイルのパス")
@click.option("-l", "--labels", type=int, default=None,
              help="条件付け用のラベル（カテゴリ番号）")
@click.option("-s", "--size", default=None,
              help="出力解像度 (例: '1024-1024')")
@click.option("-sc", "--scale_type", default="pad", show_default=True,
              help="pad, side, symm（centr, fit も可）")
@click.option("-lm", "--latmask", default=None,
              help="多重潜在ブレンディング用の外部マスクファイル（またはディレクトリ）")
# ※ オプション名は大文字指定していますが、click は内部で小文字に変換するため
@click.option("-n", "--nXY", "nxy", default="1-1", show_default=True,
              help="横×縦のフレーム分割数 (例: '1-1')")
@click.option("--splitfine", type=float, default=0.0, show_default=True,
              help="分割時のエッジシャープネス（0=滑らか、値が大きいほど細かく）")
@click.option("--splitmax", type=int, default=None,
              help="分割フレームの潜在数の上限（OOM防止用）")
@click.option("--trunc", type=float, default=0.8, show_default=True,
              help="truncation psi 0～1（低いほど安定、高いほど多様）")
@click.option("--save_lat", is_flag=True,
              help="潜在ベクトルをファイルに保存する")
@click.option("-v", "--verbose", is_flag=True,
              help="詳細出力モード")
@click.option("--noise_seed", type=int, default=3025, show_default=True,
              help="ノイズシード")
@click.option("-f", "--frames", default="200-25", show_default=True,
              help=("1補間区間における総フレーム数と補間ステップを指定します (例: '200-25')。\n"
                    "無限リアルタイム生成中でも、各補間区間で生成されるフレーム数は、\n"
                    "・補間区間の長さ、アニメーションの滑らかさや変化の速さ、\n"
                    "・平行移動や回転などの変換パラメータの生成に利用されるため、\n"
                    "ユーザーが調整できるように重要なパラメータとなっています。"))
@click.option("--cubic", is_flag=True,
              help="補間に cubic spline を使用")
@click.option("--gauss", is_flag=True,
              help="補間に Gaussian smoothing を使用")
@click.option("-at", "--anim_trans", is_flag=True,
              help="平行移動アニメーションを追加")
@click.option("-ar", "--anim_rot", is_flag=True,
              help="回転アニメーションを追加")
@click.option("-sb", "--shiftbase", type=float, default=0.0, show_default=True,
              help="タイル中心へのシフト量")
@click.option("-sm", "--shiftmax", type=float, default=0.0, show_default=True,
              help="タイル中心周りのランダムウォーク幅")
@click.option("--digress", type=float, default=0.0, show_default=True,
              help="Aydao による歪み効果の強さ")
@click.option("-as", "--affine_scale", default="1.0-1.0", show_default=True,
              help="Affine 変換用拡大率 (例: '1.0-1.0')")
@click.option("--framerate", type=int, default=30, show_default=True,
              help="ビデオのフレームレート")
@click.option("--prores", is_flag=True,
              help="ProRes 形式で動画出力")
@click.option("--variations", type=int, default=1, show_default=True,
              help="バリエーション数")
@click.option("--colab_demo", is_flag=True,
              help="Colab 上でサンプル動作するモード")
@click.option("--method", type=click.Choice(['smooth', 'random_walk']), default="smooth", show_default=True,
              help="無限生成方式: smooth は latent_anima、random_walk は各フレームに乱数を追加")
def main(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax, trunc, save_lat,
         verbose, noise_seed, frames, cubic, gauss, anim_trans, anim_rot, shiftbase, shiftmax, digress,
         affine_scale, framerate, prores, variations, colab_demo, method):
    """
    Customized StyleGAN3 on PyTorch
    (リアルタイムプレビュー & Colab デモ版)
    """
    # size: 文字列 "幅-高さ" をリスト [高さ, 幅] に変換（1つの数値なら両方同じ値）
    if size is not None:
        size = [int(s) for s in size.split('-')][::-1]
        if len(size) == 1:
            size = size * 2

    # affine_scale: "scale_y-scale_x" をリスト [scale_x, scale_y] に変換
    if affine_scale is not None:
        affine_scale = [float(s) for s in affine_scale.split('-')][::-1]

    # frames: "総フレーム数-補間ステップ" を分解
    frames_split = frames.split('-')
    if len(frames_split) != 2:
        raise click.ClickException("Invalid format for --frames. Expected format 'total_frames-fstep'.")
    frames_val, fstep = [int(s) for s in frames_split]

    # 引数をまとめた名前空間を生成（もともとの argparse の namespace と同様に）
    a = SimpleNamespace(
        out_dir=out_dir,
        model=model,
        labels=labels,
        size=size,
        scale_type=scale_type,
        latmask=latmask,
        nxy=nxy,
        splitfine=splitfine,
        splitmax=splitmax,
        trunc=trunc,
        save_lat=save_lat,
        verbose=verbose,
        noise_seed=noise_seed,
        frames=frames_val,
        fstep=fstep,
        cubic=cubic,
        gauss=gauss,
        anim_trans=anim_trans,
        anim_rot=anim_rot,
        shiftbase=shiftbase,
        shiftmax=shiftmax,
        digress=digress,
        affine_scale=affine_scale,
        framerate=framerate,
        prores=prores,
        variations=variations,
        colab_demo=colab_demo,
        method=method
    )

    if a.colab_demo:
        click.echo("Colabデモモードで起動します (cv2によるリアルタイムウィンドウは使用しません)")
        for i in range(a.variations):
            generate_colab_demo(a, a.noise_seed + i)
    else:
        click.echo("ローカル環境でのリアルタイムプレビューを行います (cv2使用)")
        for i in range(a.variations):
            generate_realtime_local(a, a.noise_seed + i)

###############################################################################
# 以下、もとの関数群
###############################################################################

def img_resize_for_cv2(img):
    """
    OpenCVウィンドウに表示するときに大きすぎる場合があるので、
    ウィンドウに収まるように必要なら縮小するための簡易関数。
    """
    max_w = 1920
    max_h = 1080
    h, w, c = img.shape
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def make_out_name(a):
    def fmt_f(v):
        return str(v).replace('.', '_')

    model_name = basename(a.model)
    out_name = f"{model_name}_seed{a.noise_seed}"
    if a.size is not None:
        out_name += f"_size{a.size[1]}x{a.size[0]}"
    out_name += f"_nXY{a.nxy}"
    out_name += f"_frames{a.frames}"
    out_name += f"_trunc{fmt_f(a.trunc)}"
    if a.cubic:
        out_name += "_cubic"
    if a.gauss:
        out_name += "_gauss"
    if a.anim_trans:
        out_name += "_at"
    if a.anim_rot:
        out_name += "_ar"
    out_name += f"_sb{fmt_f(a.shiftbase)}"
    out_name += f"_sm{fmt_f(a.shiftmax)}"
    out_name += f"_digress{fmt_f(a.digress)}"
    if a.affine_scale is not None and a.affine_scale != [1.0, 1.0]:
        out_name += "_affine"
        out_name += f"_s{fmt_f(a.affine_scale[0])}-{fmt_f(a.affine_scale[1])}"
    out_name += f"_fps{a.framerate}"
    return out_name

def checkout(output, i, out_dir):
    """
    1枚ずつファイルに保存したい場合に使う関数。
    """
    ext = 'png' if output.shape[3] == 4 else 'jpg'
    filename = osp.join(out_dir, "%06d.%s" % (i, ext))
    imsave(filename, output[0], quality=95)

def infinite_latent_smooth(z_dim, device, cubic=False, gauss=False, seed=None,
                           chunk_size=30, uniform=False):
    """
    latent_anima を使って、2つの潜在ベクトル間の補間フレームを chunk_size 生成し、
    次の区間に移るときに新たな潜在ベクトルを用意して無限に yield するジェネレータ。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    lat1 = rng.randn(z_dim)
    while True:
        lat2 = rng.randn(z_dim)
        key_latents = np.stack([lat1, lat2], axis=0)  # (2, z_dim)
        latents_np = latent_anima(
            shape=(z_dim,),
            frames=chunk_size,
            transit=chunk_size,
            key_latents=key_latents,
            somehot=None,
            smooth=0.5,
            uniform=uniform,
            cubic=cubic,
            gauss=gauss,
            seed=None,
            start_lat=None,
            loop=False,
            verbose=False
        )  # shape=(chunk_size, z_dim)
        for i in range(len(latents_np)):
            yield torch.from_numpy(latents_np[i]).unsqueeze(0).to(device)
        lat1 = lat2

def infinite_latent_random_walk(z_dim, device, seed=None, step_size=0.02):
    """
    前回の潜在ベクトルに毎フレーム少量の乱数を加えることでランダムウォークを行うジェネレータ。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    z_prev = rng.randn(z_dim)
    while True:
        z_prev = z_prev + rng.randn(z_dim) * step_size
        yield torch.from_numpy(z_prev).unsqueeze(0).to(device)

def generate_realtime_local(a, noise_seed):
    """
    無限リアルタイム生成と OpenCV による表示を行う関数。
    --method で 'smooth'（latent_anima を利用）か 'random_walk' を選択します。
    """
    import torch, numpy as np, random, os, os.path as osp, cv2, time, sys, queue, threading
    from util.utilgan import img_read, img_list, latent_anima, basename
    import dnnlib, legacy

    # シード設定
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)

    # ネットワーク読み込み用引数
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type

    # --- マスク(lmask)の設定 ---
    if a.latmask is None:
        nHW = [int(s) for s in a.nxy.split('-')][::-1]
        if len(nHW) != 2:
            raise ValueError(f"Wrong count nXY: {len(nHW)} (must be 2)")
        n_mult = nHW[0] * nHW[1]
        if a.splitmax is not None:
            n_mult = min(n_mult, a.splitmax)
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        if a.splitmax is not None:
            Gs_kwargs.splitmax = a.splitmax
        if a.verbose and n_mult > 1:
            print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = [None]
    else:
        n_mult = 2
        nHW = [1, 1]
        if osp.isfile(a.latmask):
            lmask = np.asarray([[img_read(a.latmask)[:, :, 0] / 255.]])
        elif osp.isdir(a.latmask):
            lmask = np.expand_dims(np.asarray([img_read(f)[:, :, 0] / 255. for f in img_list(a.latmask)]), 1)
        else:
            print(' !! Blending mask not found:', a.latmask)
            exit(1)
        if a.verbose:
            print(' Latent blending with mask', a.latmask, lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1)
        lmask = torch.from_numpy(lmask).to(device)
    # --- マスク設定ここまで ---

    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))

    rot = True if ('-r-' in a.model.lower() or 'sg3r-' in a.model.lower()) else False
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim

    if c_dim > 0 and a.labels is not None:
        label = torch.zeros([1, c_dim], device=device)
        label_idx = min(int(a.labels), c_dim - 1)
        label[0, label_idx] = 1
    else:
        label = None

    # --- 初回ウォームアップ推論 ---
    with torch.no_grad():
        if custom and hasattr(Gs.synthesis, 'input'):
            # カスタムモデルの場合は位置引数で渡す
            _ = Gs(
                torch.randn([1, z_dim], device=device),
                label,
                lmask[0],
                (torch.zeros([1,2], device=device),
                 torch.zeros([1,1], device=device),
                 torch.ones([1,2], device=device)),
                torch.zeros(1, device=device),  # ここは元コードに合わせた dconst[0] の値
                noise_mode='const'
            )
        else:
            _ = Gs(
                torch.randn([1, z_dim], device=device),
                label,
                lmask[0],
                truncation_psi=a.trunc,
                noise_mode='const'
            )
    # --- 初回ウォームアップここまで ---

    frame_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    # 潜在ベクトル生成（関数は省略、元コードの infinite_latent_smooth / random_walk を使用）
    if a.method == 'random_walk':
        print("=== Real-time Preview (random_walk mode) ===")
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=a.noise_seed, step_size=0.02)
    else:
        print("=== Real-time Preview (smooth latent_anima mode) ===")
        latent_gen = infinite_latent_smooth(z_dim=z_dim, device=device, cubic=a.cubic, gauss=a.gauss, seed=a.noise_seed, chunk_size=60, uniform=False)

    def producer_thread():
        frame_idx_local = 0
        while not stop_event.is_set():
            z_current = next(latent_gen)
            if lmask[0] is None:
                latmask_current = None
            else:
                latmask_current = lmask[frame_idx_local % len(lmask)]
            # ここでは dconst_current はダミー値として torch.zeros を使用（元コードに合わせて適宜変更してください）
            dconst_current = torch.zeros(1, device=device)
            if custom and hasattr(Gs.synthesis, 'input'):
                trans_param = (
                    torch.zeros([1, 2], device=device),
                    torch.zeros([1, 1], device=device),
                    torch.ones([1, 2], device=device)
                )
            else:
                trans_param = None
            with torch.no_grad():
                if custom and hasattr(Gs.synthesis, 'input'):
                    out = Gs(
                        z_current,
                        label,
                        latmask_current,
                        trans_param,
                        dconst_current,
                        noise_mode='const'
                    )
                else:
                    out = Gs(
                        z_current,
                        label,
                        latmask_current,
                        truncation_psi=a.trunc,
                        noise_mode='const'
                    )
            out = (out.permute(0,2,3,1) * 127.5 + 128).clamp(0,255).to(torch.uint8)
            out_np = out[0].cpu().numpy()[..., ::-1]
            frame_queue.put(out_np)
            frame_idx_local += 1

    thread_prod = threading.Thread(target=producer_thread, daemon=True)
    thread_prod.start()

    print("ウィンドウが表示されます。終了する場合は 'q' キーを押してください。")
    fps_count = 0
    t0 = time.time()
    while True:
        out_cv = frame_queue.get()
        out_cv = img_resize_for_cv2(out_cv)
        cv2.imshow("StyleGAN3 Real-time Preview", out_cv)
        fps_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"\r{fps_count / elapsed:.2f} fps", end='')
            sys.stdout.flush()
            t0 = time.time()
            fps_count = 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n終了します。")
            stop_event.set()
            break
    cv2.destroyAllWindows()

def generate_colab_demo(a, noise_seed):
    """
    Colab 上で短いループを回し、画像をノートブックセルに表示するサンプルモード。
    """
    print("=== Colab デモ開始 ===")
    print("(こちらは従来のフレーム固定デモです)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pkl_name = osp.splitext(a.model)[0]
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f)['G_ema'].to(device)
    frames = 30
    for i in range(frames):
        z = torch.randn([1, Gs.z_dim], device=device)
        with torch.no_grad():
            output = Gs(z, None, truncation_psi=a.trunc, noise_mode='const')
        output = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_np = output[0].cpu().numpy()
        clear_output(wait=True)
        display(Image.fromarray(out_np, 'RGB'))
        time.sleep(0.2)
    print("=== Colab デモ終了 ===")

if __name__ == '__main__':
    main()
