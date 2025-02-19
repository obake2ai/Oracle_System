# SSH経由で転送する先の設定

SSH_CONFIG = {
    # 共通設定
    'port': 22,
    'username': 'pi',
    'password': 'raspberry',
    # 各転送先の設定（ここに最大20件程度追加可能）
    'destinations': [
        {
            'host': '192.168.10.33', #zero2wh04.local
            'local_dir': 'outputs/12x6',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/samples'
        },
        {
            'host': '192.168.10.29', #zero2wh05.local
            'local_dir': 'outputs/12x6',
            'remote_dir': '/home/pi/sshtest'
        },
        {
            'host': '192.168.10.26', #zero2wh06.local
            'local_dir': 'outputs/12x6',         
            'remote_dir': '/home/pi/sshtest'
        },
    ]
}
