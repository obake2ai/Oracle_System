#!/usr/bin/env python3
import os
import glob
import time
import paramiko
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
from config.config import GEN_CONFIG
from config.ssh import SSH_CONFIG

def get_target_size(local_dir):
    """
    local_dir のベースネームが "NxM" または "NxM-xxx" の形式の場合、
    ハイフン以降を無視して "NxM" 部分だけから (N×64, M×64) のサイズを返す。
    例: "6x12"    -> (384, 768)
         "6x12-a"  -> (384, 768)
    形式に合致しなければ None を返す。
    """
    base = os.path.basename(os.path.normpath(local_dir))
    base = base.split('-')[0]
    if "x" in base:
        parts = base.split("x")
        try:
            cols = int(parts[0])
            rows = int(parts[1])
            return (cols * 64, rows * 64)
        except Exception:
            return None
    return None

def load_and_resize_image(filepath, target_size):
    """
    filepath の画像を開き、target_size が指定されていればリサイズして返す。
    """
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        return img.copy()

def generate_transition_frames_pixelwise(img1, img2, num_frames):
    """
    img1, img2 は PIL.Image 形式、num_frames は総フレーム数。
    各ピクセルごとに、ランダムな閾値を用いて、img1 のピクセルが
    あるタイミングから img2 のピクセルに切り替わる中間フレームを生成する。
    戻り値は num_frames-1 個の中間フレーム（最終フレームは new_img とする）。
    """
    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)
    H, W, C = arr1.shape
    threshold = np.random.rand(H, W)  # 各ピクセルの切替タイミング (0～1)

    frames = []
    for i in range(1, num_frames):
        progress = i / num_frames
        mask = (progress >= threshold)
        mask3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        frame_arr = np.where(mask3, arr2, arr1)
        frame = Image.fromarray(frame_arr.astype(np.uint8))
        frames.append(frame)
    return frames

def send_image_via_sftp(ssh, image, remote_path):
    """
    PIL.Image を PNG 形式に変換し、BytesIO 経由で sftp.putfo で送信する。
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    sftp = ssh.open_sftp()
    sftp.putfo(buf, remote_path)
    sftp.close()
    buf.close()

def save_preview_to_log(local_dir, image, preview_filename):
    """
    preview_image を、local_dir/log フォルダに preview_filename で保存する。
    画像は圧縮済み（リサイズ済み）のものを保存します。
    """
    log_folder = os.path.join(local_dir, "log")
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, preview_filename)
    image.save(log_path, format="PNG")
    print(f"Preview saved to log: {log_path}")

def transfer_transition_sequence(ssh, remote_dir, target_size, local_dir, prev_img, new_img, prev_filename, new_filename):
    """
    prev_img から new_img へのトランジションを生成し、指定の間隔内に順次送信します。
    各フレームはファイル名に frame 番号を含み、ローカルの log フォルダにも保存されます。
    最後に new_img を送信し、keep 秒間保持します。
    """
    total_interval = GEN_CONFIG.get("interval", 300)              # 総トランジション時間（秒）
    transition_interval = GEN_CONFIG.get("transition_interval", 1)  # 各中間フレームの送信間隔（秒）
    keep_time = GEN_CONFIG.get("keep", 5)                           # 最終画像の保持時間（秒）
    num_frames = max(1, int(total_interval / transition_interval))

    print(f"Generating {num_frames} pixelwise transition frames.")
    frames = generate_transition_frames_pixelwise(prev_img, new_img, num_frames)

    for idx, frame in enumerate(frames):
        preview_filename = f"{prev_filename}_{new_filename}_frame{idx+1:03d}.png"
        remote_preview_path = os.path.join(remote_dir, preview_filename)
        print(f"Sending transition frame {idx+1}/{len(frames)} as {preview_filename} to {remote_preview_path} ...")
        try:
            send_image_via_sftp(ssh, frame, remote_preview_path)
        except Exception as e:
            print(f"Error sending frame {idx+1}: {e}.")
        save_preview_to_log(local_dir, frame, preview_filename)
        time.sleep(transition_interval)

    final_filename = f"{prev_filename}_{new_filename}_final.png"
    remote_final_path = os.path.join(remote_dir, final_filename)
    print(f"Sending final image as {final_filename} to {remote_final_path} ...")
    try:
        send_image_via_sftp(ssh, new_img, remote_final_path)
    except Exception as e:
        print(f"Error sending final image: {e}.")
    save_preview_to_log(local_dir, new_img, final_filename)
    print(f"Holding final preview for {keep_time} seconds ...")
    time.sleep(keep_time)

def save_file_to_log(filepath):
    """
    filepath のファイルを、同じフォルダ内の log サブフォルダへ
    日時を付与したファイル名で移動保存する。
    """
    folder = os.path.dirname(filepath)
    log_folder = os.path.join(folder, "log")
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(filepath)
    new_name = f"{timestamp}_{base}"
    new_path = os.path.join(log_folder, new_name)
    os.rename(filepath, new_path)
    print(f"File moved to log: {new_path}")

def process_local_directory(local_dir, dest_list):
    """
    指定されたローカルディレクトリ内の PNG ファイルを更新時刻順に処理し、
    同じ local_dir を参照する各転送先（dest_list）へ初回画像またはトランジション画像を送信します。
    各転送先は独自の SSH 接続および前回画像状態を保持し、処理完了後に
    そのファイルは log フォルダへ移動されます。
    """
    print(f"\nProcessing local directory '{local_dir}' for {len(dest_list)} destination(s).")
    # 各転送先ごとに SSH 接続を確立し、状態を保持する
    connections = {}
    for dest in dest_list:
        host = dest.get("host")
        remote_dir = dest.get("remote_dir")
        port = SSH_CONFIG.get("port", 22)
        username = SSH_CONFIG.get("username")
        password = SSH_CONFIG.get("password")
        print(f"[{host}] Connecting to {host}:{port} as {username} ...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username=username, password=password)
        # リモートディレクトリの存在確認と作成
        sftp = ssh.open_sftp()
        try:
            sftp.stat(remote_dir)
        except IOError:
            print(f"[{host}] Remote directory '{remote_dir}' does not exist. Creating it.")
            sftp.mkdir(remote_dir)
        sftp.close()
        connections[host] = {"ssh": ssh, "prev_img": None, "prev_filename": None, "dest": dest}

    target_size = get_target_size(local_dir)
    files = sorted(glob.glob(os.path.join(local_dir, "*.png")), key=os.path.getmtime)

    for filepath in files:
        try:
            print(f"Processing file: {filepath}")
            new_img = load_and_resize_image(filepath, target_size)
            new_filename = os.path.splitext(os.path.basename(filepath))[0]
            for host, conn in connections.items():
                ssh = conn["ssh"]
                dest = conn["dest"]
                remote_dir = dest.get("remote_dir")
                if conn["prev_img"] is None:
                    # 初回画像の送信
                    remote_preview_path = os.path.join(remote_dir, f"{new_filename}_initial.png")
                    print(f"[{host}] No previous image; sending new image directly as initial.")
                    try:
                        send_image_via_sftp(ssh, new_img, remote_preview_path)
                    except Exception as e:
                        print(f"[{host}] Error sending initial image: {e}.")
                    save_preview_to_log(local_dir, new_img, f"{new_filename}_initial.png")
                    conn["prev_img"] = new_img
                    conn["prev_filename"] = new_filename
                    keep_time = GEN_CONFIG.get("keep", 5)
                    print(f"[{host}] Holding preview for {keep_time} seconds ...")
                    time.sleep(keep_time)
                else:
                    # トランジション処理
                    print(f"[{host}] Starting pixelwise transition from previous image to new image ...")
                    transfer_transition_sequence(ssh, remote_dir, target_size, local_dir,
                                                   conn["prev_img"], new_img,
                                                   conn["prev_filename"], new_filename)
                    conn["prev_img"] = new_img
                    conn["prev_filename"] = new_filename
            # 全転送先で処理が完了したら、元のファイルを log フォルダへ移動
            save_file_to_log(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}. Skipping this file.")

    # すべての SSH 接続をクローズ
    for host, conn in connections.items():
        conn["ssh"].close()
        print(f"[{host}] Processing complete and SSH connection closed.")

def main():
    check_interval = GEN_CONFIG.get("check_interval", 30)  # フォルダ監視間隔（秒）
    # 同じ local_dir を参照する転送先をグループ化する
    local_dir_to_destinations = {}
    for dest in SSH_CONFIG.get("destinations", []):
        local_dir = dest.get("local_dir")
        if local_dir not in local_dir_to_destinations:
            local_dir_to_destinations[local_dir] = []
        local_dir_to_destinations[local_dir].append(dest)

    while True:
        for local_dir, dest_list in local_dir_to_destinations.items():
            image_files = glob.glob(os.path.join(local_dir, "*.png"))
            if image_files:
                process_local_directory(local_dir, dest_list)
            else:
                hosts = [d.get("host") for d in dest_list]
                print(f"No new images in '{local_dir}' for destinations: {hosts}")
        print(f"Waiting for {check_interval} seconds before next folder check...\n")
        time.sleep(check_interval)

if __name__ == "__main__":
    main()
