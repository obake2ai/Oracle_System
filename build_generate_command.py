#!/usr/bin/env python3
import os
from config.config import GEN_CONFIG

def build_generate_image_command():
    """
    config/config.py の GEN_CONFIG をもとに、
    generate_image.py を呼び出すためのコマンドライン引数リストを構築します。
    """
    cmd = ["python", "generate_image.py"]

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

if __name__ == "__main__":
    cmd = build_generate_image_command()
    # コマンドライン引数リストを表示
    print("Generated command-line argument list:")
    print(" ".join(cmd))

    os.system(" ".join(cmd))
