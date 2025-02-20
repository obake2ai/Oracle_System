#!/usr/bin/env python3
import os
import time
import random
from config.config import GEN_CONFIG

def compute_size_from_outdir(out_dir, base_size = 128):
    """
    out_dir のベースネームから、サイズ文字列を動的に計算します。
    base_size を用いて計算するため、後で比率を変更しやすいです。
    ・ベースネームに "12x6" が含まれている場合 → f"{12*base_size}-{6*base_size}" (例: "3072-1536")
    ・ベースネームに "4x3" が含まれている場合 → f"{4*base_size}-{6*base_size}" (例: "1024-1536")
    それ以外の場合は、GEN_CONFIG に設定されている 'size' をそのまま返します。
    """
    base = os.path.basename(out_dir)
    if "12x3" in base:
        width = 3 * base_size
        height = 12 * base_size
        return f"{width}-{height}"
    elif "4x3" in base:
        width = 6 * base_size
        height = 4 * base_size
        return f"{width}-{height}"
    else:
        return GEN_CONFIG.get("size", "1024-1024")

def build_generate_image_command():
    """
    GEN_CONFIG の値をもとに、src/generate_image.py を呼び出すためのコマンドライン引数リストを構築します。
    """
    cmd = ["python", "src/generate_image.py"]

    # 基本設定
    cmd.extend(["-o", GEN_CONFIG["out_dir"]])
    cmd.extend(["-m", GEN_CONFIG["model"]])
    if GEN_CONFIG["labels"]:
        cmd.extend(["-l", GEN_CONFIG["labels"]])
    cmd.extend(["-s", GEN_CONFIG["size"]])
    cmd.extend(["-sc", GEN_CONFIG["scale_type"]])
    if GEN_CONFIG["latmask"]:
        cmd.extend(["-lm", GEN_CONFIG["latmask"]])
    cmd.extend(["-n", GEN_CONFIG["nXY"]])
    cmd.extend(["--splitfine", str(GEN_CONFIG["splitfine"])])
    if GEN_CONFIG["splitmax"] is not None:
        cmd.extend(["--splitmax", str(GEN_CONFIG["splitmax"])])
    cmd.extend(["--trunc", str(GEN_CONFIG["trunc"])])
    if GEN_CONFIG["save_lat"]:
        cmd.append("--save_lat")
    if GEN_CONFIG["verbose"]:
        cmd.append("--verbose")
    cmd.extend(["--noise_seed", str(GEN_CONFIG["noise_seed"])])

    # アニメーション関連・フレーム指定
    cmd.extend(["-f", GEN_CONFIG["frames"]])
    if GEN_CONFIG["cubic"]:
        cmd.append("--cubic")
    if GEN_CONFIG["gauss"]:
        cmd.append("--gauss")

    # SG3変形関連
    if GEN_CONFIG["anim_trans"]:
        cmd.append("--anim_trans")
    if GEN_CONFIG["anim_rot"]:
        cmd.append("--anim_rot")
    cmd.extend(["-sb", str(GEN_CONFIG["shiftbase"])])
    cmd.extend(["-sm", str(GEN_CONFIG["shiftmax"])])
    cmd.extend(["--digress", str(GEN_CONFIG["digress"])])

    # アフィン変換
    cmd.extend(["-as", GEN_CONFIG["affine_scale"]])

    # 動画設定
    cmd.extend(["--framerate", str(GEN_CONFIG["framerate"])])
    if GEN_CONFIG["prores"]:
        cmd.append("--prores")
    cmd.extend(["--variations", str(GEN_CONFIG["variations"])])

    # 画像出力モード
    if GEN_CONFIG["image"]:
        cmd.append("--image")

    return cmd

def main():
    # 複数の保存先ディレクトリ（各保存先で別々の画像生成を行う）
    out_dirs = ['outputs/12x3-A', 'outputs/12x3-B']

    for out_dir in out_dirs:
        # 動的に GEN_CONFIG を更新
        GEN_CONFIG["out_dir"] = out_dir
        GEN_CONFIG["size"] = compute_size_from_outdir(out_dir)
        # ノイズシードは毎回ランダムに（例：1000〜10000）
        GEN_CONFIG["noise_seed"] = random.randint(1000, 10000)

        cmd = build_generate_image_command()
        print(f"\n--- Generating images for '{out_dir}' ---")
        print("Command:", " ".join(cmd))
        os.system(" ".join(cmd))

        # 次の生成まで少し待機（必要に応じて調整）
        time.sleep(30)

if __name__ == "__main__":
    main()
