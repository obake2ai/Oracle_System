#B1F
STYLEGAN_CONFIG = {
    "out_dir": "_out",
    "model": "models/embryo-stylegan3-r-network-snapshot-000096",
    "labels": None,
    "size": "2058-920",            # 出力解像度 2058-920:8fps 1610-720:14fps
    "scale_type": "pad",
    "latmask": None,
    "nXY": "1-1",                # latent blending 用：フレームの分割数（例："1-1"）
    "splitfine": 0.0,            # 分割時のエッジシャープネス（0～）
    "splitmax": None,            # 分割時の最大 latent 数（OOM 回避用）
    "trunc": 1.0,                # truncation psi
    "save_lat": False,           # latent 保存フラグ
    "verbose": False,
    "noise_seed": 3036,
    "sg_gpu": "cuda:0",
    # アニメーション関連
    "frames": "400-400",          # （未使用だが互換性のため）
    "cubic": False,
    "gauss": False,
    # SG3 の変換（アニメーション）パラメータ
    "anim_trans": False,
    "anim_rot": False,
    "shiftbase": 0.5,
    "shiftmax": 0.2,
    "digress": -10,
    # Affine Conversion（拡大縮小）
    "affine_scale": "0.7-0.7",
    # 動画保存などの設定
    "framerate": 30,
    "prores": False,
    "variations": 1,
    # 無限リアルタイム生成の方式
    "method": "smooth",       #"smooth" or "random_walk"
    "chunk_size": 40000,           #大きいほど変化がゆっくりに for smooth
    "random_step_size": 0.0001,     #大きいほど変化がゆっくりに for random
    # ---- 以下、GPT 関連の設定 ----
    "gpt_model": "./models/gpt_model_epoch_16000.pth",
    "gpt_prompt": "I'm praying: ",
    "max_new_tokens": 20,
    "context_length": 512,
    "gpt_gpu": "cuda:0",
    # ---- 以下、オーバーレイテキスト用フォント設定 ----
    "font_path_ja": "data/fonts/FOT-TsukuGoPr5-D.otf",
    "font_path_en": "data/fonts/Acumin-RPro.otf",
    "default_font_scale": 0.5,
    "default_font_thickness": 0.5,
    "font_color": (255, 255, 255),
    "subtitle_ja_font_size": 21,  # 日本語字幕のフォントサイズ（pt）
    "subtitle_en_font_size": 24,  # 英語字幕のフォントサイズ（pt）
    "subtitle_ja_y": 5,          # 日本語字幕の開始位置（上端から 10% の位置）
    "subtitle_en_y": 95,          # 英語字幕の開始位置（上端から 80% の位置）
    # ---- スライドショーの設定 ----
    "display_time": 11,
    "clear_time": 1.0
}

#1F
GEN_CONFIG = {
    # 生成画像の出力先フォルダ
    'out_dir': 'outputs/12x3',
    # 使用するモデルのパス（_genSGAN3.py内で読み込まれるpklファイル）
    'model': 'models/embryo-stylegan3-r-network-snapshot-000096',
    # ラベル指定（例："1-7-4" など、条件付け用）
    'labels': None,
    # 画像サイズ（例："1024-1024" ※内部で [1024,1024] に変換）
    'size': '1024-1024',
    'scale_type': 'pad',
    'latmask': None,
    'nXY': '1-1',
    'splitfine': 0.0,
    'splitmax': None,
    # truncation psi 値
    'trunc': 1.0,
    'save_lat': False,
    'verbose': False,
    # 初期乱数シード
    'noise_seed': 3029,
    # フレーム数と補間ステップ（ここでは画像出力モードなので "1-1" ＝1フレーム）
    'frames': '1-1',
    'cubic': False,
    'gauss': False,
    # SG3変形関連
    'anim_trans': False,
    'anim_rot': False,
    'shiftbase': 0.0,
    'shiftmax': 0.0,
    'digress': 0.0,
    # アフィン変換（縦横のスケールファクタ）
    'affine_scale': '1.0-1.0',
    # 動画設定（画像出力モードの場合はあまり影響しない）
    'framerate': 30,
    'prores': False,
    # 生成するバリエーション数（シードを順次更新）
    'variations': 1,
    # 画像出力モードにする（動画ではなく各フレームを個別画像として保存）
    'image': True,
    # 画像生成の実行間隔（秒）
    'ssh_check_interval': 60,
    'transition_interval': 180,
    'generate_interval': 600
}
