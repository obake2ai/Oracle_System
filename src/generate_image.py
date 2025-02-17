import os
import click
import _genSGAN3  # ファイル配置に合わせてパスを調整してください
from config.config import STYLEGAN_CONFIG

@click.command()
# 基本設定
@click.option("-o", "--out_dir", default="_out", help="出力先ディレクトリ")
@click.option("-m", "--model", default="models/ffhq1024.pkl", help="pklチェックポイントのパス")
@click.option("-l", "--labels", default=None, help="条件付け用のラベル/カテゴリ（例: 1-7-4）")
# カスタムオプション
@click.option("-s", "--size", default="1024-1024", help="出力解像度（例: 1024-1024）")
@click.option("-sc", "--scale_type", default="pad", help="pad, side, symm (centr, fit も可)")
@click.option("-lm", "--latmask", default=None, help="複数latentのブレンド用外部マスクファイル（またはディレクトリ）")
@click.option("-n", "--nXY", default="1-1", help="フレーム分割数（幅×高さ、例: 1-1）")
@click.option("--splitfine", default=0.0, type=float, help="分割時のエッジシャープネス（0で滑らか、値が大きいと細かく）")
@click.option("--splitmax", default=None, type=int, help="分割時のlatent最大数（OOM防止用）")
@click.option("--trunc", default=0.8, type=float, help="truncation psi (0..1、低いほど安定・高いほど多様)")
@click.option("--save_lat", is_flag=True, help="latentベクトルをファイルに保存")
@click.option("-v", "--verbose", is_flag=True, help="詳細情報の表示")
@click.option("--noise_seed", default=3025, type=int, help="乱数シード")
# アニメーション関連
@click.option("-f", "--frames", default="200-25", help="生成フレーム数と補間ステップ（例: 200-25）")
@click.option("--cubic", is_flag=True, help="平滑化に3次スプラインを使用")
@click.option("--gauss", is_flag=True, help="平滑化にGaussianフィルタを使用")
# SG3変形
@click.option("-at", "--anim_trans", is_flag=True, help="平行移動アニメーションを追加")
@click.option("-ar", "--anim_rot", is_flag=True, help="回転アニメーションを追加")
@click.option("-sb", "--shiftbase", default=0.0, type=float, help="タイル中心へのシフト量")
@click.option("-sm", "--shiftmax", default=0.0, type=float, help="タイル中心周りのランダムウォーク量")
@click.option("--digress", default=0.0, type=float, help="Aydaoによる歪み効果の強さ")
# アフィン変換
@click.option("-as", "--affine_scale", default="1.0-1.0", help="縦横のスケールファクタ（例: 1.0-1.0）")
# 動画設定
@click.option("--framerate", default=30, type=int, help="フレームレート")
@click.option("--prores", is_flag=True, help="ProRes形式で動画出力")
@click.option("--variations", default=1, type=int, help="バリエーション数")
# 画像出力モード
@click.option("--image", is_flag=True, help="動画ではなく各フレームを個別画像として保存")
def main(out_dir, model, labels, size, scale_type, latmask, nxy, splitfine, splitmax,
         trunc, save_lat, verbose, noise_seed, frames, cubic, gauss,
         anim_trans, anim_rot, shiftbase, shiftmax, digress, affine_scale,
         framerate, prores, variations, image):
    """
    各種オプションを指定して、SGAN3 による画像/動画生成を実行する。
    """
    # 各オプションを _genSGAN3 のグローバル変数 a に反映
    _genSGAN3.a.out_dir      = out_dir
    _genSGAN3.a.model        = model
    _genSGAN3.a.labels       = labels
    _genSGAN3.a.size         = [int(s) for s in size.split("-")] if size is not None else None
    _genSGAN3.a.scale_type   = scale_type
    _genSGAN3.a.latmask      = latmask
    _genSGAN3.a.nXY          = nxy
    _genSGAN3.a.splitfine    = splitfine
    _genSGAN3.a.splitmax     = splitmax
    _genSGAN3.a.trunc        = trunc
    _genSGAN3.a.save_lat     = save_lat
    _genSGAN3.a.verbose      = verbose
    _genSGAN3.a.noise_seed   = noise_seed
    _genSGAN3.a.frames       = frames
    _genSGAN3.a.cubic        = cubic
    _genSGAN3.a.gauss        = gauss
    _genSGAN3.a.anim_trans   = anim_trans
    _genSGAN3.a.anim_rot     = anim_rot
    _genSGAN3.a.shiftbase    = shiftbase
    _genSGAN3.a.shiftmax     = shiftmax
    _genSGAN3.a.digress      = digress
    _genSGAN3.a.affine_scale = [float(s) for s in affine_scale.split("-")]
    _genSGAN3.a.framerate    = framerate
    _genSGAN3.a.prores       = prores
    _genSGAN3.a.variations   = variations
    _genSGAN3.a.image        = image

    os.makedirs(out_dir, exist_ok=True)

    click.echo("生成開始 ---------------------")
    click.echo(f"出力先: {out_dir}")
    click.echo(f"モデル: {model}")
    click.echo(f"その他パラメータ: frames={frames}, size={size}, noise_seed={noise_seed}, variations={variations}")

    # variations (複数バリエーションの場合、シードを変えて生成)
    for i in range(variations):
        current_seed = noise_seed + i
        click.echo(f"\nVariation {i+1}/{variations} : seed {current_seed}")
        _genSGAN3.generate(current_seed)

    click.echo("\nすべての生成が完了しました。")

if __name__ == "__main__":
    main()
