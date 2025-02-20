# SSH経由で転送する先の設定

SSH_CONFIG = {
    # 共通設定
    'port': 22,
    'username': 'pi',
    'password': 'raspberry',
    'connection_timeout': 3,
    # 各転送先の設定（ここに最大20件程度追加可能）
    'destinations': [
        {
            'host': '192.168.10.33',
            'hostname': 'zero2wh07.local',
            'local_dir': 'outputs/12x3-A',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
        {
            'host': '192.168.10.29',
            'hostname': 'zero2wh05.local',
            'local_dir': 'outputs/12x3-A',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
        {
            'host': '192.168.10.26',
            'hostname': 'zero2wh06.local',
            'local_dir': 'outputs/12x3-A',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
        {
            'host': '192.168.10.35',
            'hostname': 'zero2wh01.local',
            'local_dir': 'outputs/12x3-B',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
        {
            'host': '192.168.10.34',
            'hostname': 'zero2wh02.local',
            'local_dir': 'outputs/12x3-A',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
        {
            'host': '192.168.10.22',
            'hostname': 'zero2wh03.local',
            'local_dir': 'outputs/12x3-A',
            'remote_dir': '/home/pi/Oracle_LEDmatrix/share/12x3'
        },
    ]
}
