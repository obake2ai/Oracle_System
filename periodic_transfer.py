#!/usr/bin/env python3
import os
import time
import glob
import subprocess
import paramiko
from config.config import GEN_CONFIG
from config.ssh import SSH_CONFIG

def build_generate_image_command():
    """
    config/config.py の GEN_CONFIG の値から、
    generate_image.py を呼び出すためのコマンドライン引数リストを構築する。
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

def transfer_files_via_ssh(local_dir, remote_dir):
    """
    Paramiko を使って local_dir 内の PNG ファイルを remote_dir へ転送する。
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    host = SSH_CONFIG["host"]
    port = SSH_CONFIG.get("port", 22)
    username = SSH_CONFIG["username"]
    password = SSH_CONFIG["password"]

    print(f"SSH 接続: {host}:{port} as {username}")
    ssh.connect(host, port=port, username=username, password=password)

    sftp = ssh.open_sftp()
    try:
        sftp.stat(remote_dir)
    except IOError:
        print(f"リモートディレクトリ {remote_dir} が存在しないため作成します。")
        sftp.mkdir(remote_dir)

    image_files = glob.glob(os.path.join(local_dir, "*.png"))
    for filepath in image_files:
        filename = os.path.basename(filepath)
        remote_path = os.path.join(remote_dir, filename)
        print(f"{filename} を転送中: {remote_path}")
        sftp.put(filepath, remote_path)

    sftp.close()
    ssh.close()
    print("SSH 転送完了。")

def main():
    out_dir = GEN_CONFIG.get("out_dir", "_out")
    interval = GEN_CONFIG.get("interval", 300)  # 生成間隔（秒）

    while True:
        # 生成コマンド構築
        cmd = build_generate_image_command()
        print("実行コマンド:", " ".join(cmd))

        # generate_image.py を実行
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("generate_image.py の実行に失敗しました:", e)
            break

        # 画像生成完了後、書き込み完了待ちのため少し待機
        time.sleep(2)

        # SSH 転送実行
        remote_dir = SSH_CONFIG["remote_dir"]
        print("画像転送開始 ---------------------")
        transfer_files_via_ssh(out_dir, remote_dir)

        # 転送後、ローカル画像を削除（重複転送防止）
        image_files = glob.glob(os.path.join(out_dir, "*.png"))
        for f in image_files:
            os.remove(f)
        print("ローカルの画像を削除しました。")

        print(f"次回生成まで {interval} 秒待機します...\n")
        time.sleep(interval)

if __name__ == "__main__":
    main()
