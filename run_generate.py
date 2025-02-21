#!/usr/bin/env python3
import os
import time
from config.config import GEN_CONFIG

def main():
    # GEN_CONFIG から実行間隔（秒）を取得（キーが存在しない場合は 60 秒をデフォルトとする）
    time_interval = GEN_CONFIG.get("generate_interval", 60)

    while True:
        print("build_generate_command.py を実行します...")
        # build_generate_command.py を実行（同じディレクトリにある前提）
        os.system("python build_generate_command.py")
        print(f"{time_interval} 秒後に再実行します...")
        time.sleep(time_interval)

if __name__ == "__main__":
    main()
