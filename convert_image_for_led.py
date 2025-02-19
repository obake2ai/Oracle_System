import click
from PIL import Image
import os

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--led-rows', default=64, help='Number of LED rows per panel')
@click.option('--led-cols', default=64, help='Number of LED columns per panel')
@click.option('--led-chain', default=12, help='Number of panels chained horizontally')
@click.option('--led-parallel', default=6, help='Number of parallel chains of panels')
def convert_image_for_led_matrix(input_path, output_dir, led_rows, led_cols, led_chain, led_parallel):
    # 設定に基づく最終的な画像サイズ
    final_width = led_cols * led_chain  # 64 x 12 = 768
    final_height = led_rows * led_parallel  # 64 x 6 = 384

    # 画像を開いてリサイズ
    image = Image.open(input_path)
    image = image.resize((final_width, final_height), Image.LANCZOS)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ファイル名（拡張子なし）を取得
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 6つの分割画像を保存
    for i in range(led_parallel):
        y_offset = i * led_rows
        cropped_image = image.crop((0, y_offset, final_width, y_offset + led_rows))
        output_path = os.path.join(output_dir, f"{base_name}_{i}.ppm")
        cropped_image.save(output_path, format="PPM")
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    convert_image_for_led_matrix()
