#!/usr/bin/env python3
import os
import glob
import time
import paramiko
from config.config import GEN_CONFIG
from config.ssh import SSH_CONFIG

def transfer_files_via_ssh(local_dir, remote_dir):
    """
    Paramiko を用いて、local_dir 内の PNG ファイルを remote_dir へ転送します。
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        SSH_CONFIG["host"],
        port=SSH_CONFIG.get("port", 22),
        username=SSH_CONFIG["username"],
        password=SSH_CONFIG["password"]
    )
    sftp = ssh.open_sftp()
    # リモートディレクトリの存在チェック（なければ作成）
    try:
        sftp.stat(remote_dir)
    except IOError:
        print(f"Remote directory '{remote_dir}' does not exist. Creating it.")
        sftp.mkdir(remote_dir)

    image_files = glob.glob(os.path.join(local_dir, "*.png"))
    for filepath in image_files:
        filename = os.path.basename(filepath)
        remote_path = os.path.join(remote_dir, filename)
        print(f"Transferring {filename} to {remote_path}...")
        sftp.put(filepath, remote_path)

    sftp.close()
    ssh.close()
    print("SSH transfer complete.")

def main():
    local_dir = GEN_CONFIG.get("out_dir", "_out")
    remote_dir = SSH_CONFIG["remote_dir"]
    transfer_interval = GEN_CONFIG.get("interval", 300)  # 生成間隔（秒）

    while True:
        print("Checking for new images in:", local_dir)
        image_files = glob.glob(os.path.join(local_dir, "*.png"))
        if image_files:
            print(f"Found {len(image_files)} image(s). Starting transfer.")
            transfer_files_via_ssh(local_dir, remote_dir)
            # 転送後、ローカルの画像を削除（重複転送防止）
            for f in image_files:
                os.remove(f)
            print("Local images deleted after transfer.")
        else:
            print("No images found.")

        print(f"Waiting for {transfer_interval} seconds before next check...\n")
        time.sleep(transfer_interval)

if __name__ == "__main__":
    main()
