STYLEGAN_CONFIG = {
    "out_dir": "_out",
    "model": "models/embryo-stylegan3-r-network-snapshot-000096",
    "labels": None,
    "size": "1920-708",            # 出力解像度 (後で [height, width] に変換)
    "scale_type": "pad",
    "latmask": None,
    "nXY": "1-1",                # latent blending 用：フレームの分割数（例："1-1"）
    "splitfine": 0.0,            # 分割時のエッジシャープネス（0～）
    "splitmax": None,            # 分割時の最大 latent 数（OOM 回避用）
    "trunc": 0.9,                # truncation psi
    "save_lat": False,           # latent 保存フラグ
    "verbose": False,
    "noise_seed": 3025,
    "sg_gpu": "cuda:0",
    # アニメーション関連
    "frames": "240-120",          # （未使用だが互換性のため）
    "cubic": False,
    "gauss": False,
    # SG3 の変換（アニメーション）パラメータ
    "anim_trans": True,
    "anim_rot": True,
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
    "step_size": 0.0001,
    # ---- 以下、GPT 関連の設定 ----
    "gpt_model": "./models/gpt_model_epoch_16000.pth",
    "gpt_prompt": "I'm praying: ",
    "max_new_tokens": 40,
    "context_length": 512,
    "gpt_gpu": "cuda:0",
    # ---- 以下、オーバーレイテキスト用フォント設定 ----
    "font_path": "data/fonts/NotoSansCJK-Regular.ttc",
    "default_font_scale": 0.5,
    "default_font_thickness": 2,
    "font_color": (255, 255, 255),
    # ---- スライドショーの設定 ----
    "display_time": 10,
    "clear_time": 0.5
}
