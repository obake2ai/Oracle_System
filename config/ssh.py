# SSH経由で転送する先の設定

SSH_CONFIG = {
    # 転送先ホストのIPまたはホスト名
    'host': 'zero2wh16.local',
    # SSHポート（通常は22）
    'port': 22,
    # ログインユーザ名
    'username': 'pi',
    # パスワード（鍵認証の場合はパスワード不要。または key_filename を使う）
    'password': 'raspberry',
    # 転送先のリモートフォルダパス
    'remote_dir': '/home/pi/sshtest',
}
