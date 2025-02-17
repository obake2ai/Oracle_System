# SSH経由で転送する先の設定

SSH_CONFIG = {
    # 共通設定
    'port': 22,
    'username': 'pi',
    'password': 'raspberry',
    # 各転送先の設定（ここに最大20件程度追加可能）
    'destinations': [
        {
            'host': 'zero2wh16.local',
            'local_dir': 'outputs/12x6',           # 例：画像生成スクリプトの出力フォルダ
            'remote_dir': '/home/pi/sshtest'
        },
        {
            'host': 'zero2wh15.local',
            'local_dir': 'outputs/4x3-A',           # 例：画像生成スクリプトの出力フォルダ
            'remote_dir': '/home/pi/sshtest'
        },
    ]
}
