import torch
import click
import tiktoken
from util.llm import GPT

@click.command()
@click.option(
    '--model-path',
    default="./models/gpt_model_epoch_16000.pth",
    show_default=True,
    help="モデルのチェックポイントファイルのパス"
)
@click.option(
    '--prompt',
    default="I'm praying: ",
    show_default=True,
    help="生成の起点となるプロンプト"
)
@click.option(
    '--max-new-tokens',
    default=500,
    show_default=True,
    type=int,
    help="生成する最大トークン数"
)
@click.option(
    '--context-length',
    default=512,
    show_default=True,
    type=int,
    help="モデルのコンテキスト長"
)
@click.option(
    '--device',
    default="cuda",
    show_default=True,
    help="推論に使用するデバイス（'cuda'または'cpu'）"
)
def generate_sample(model_path, prompt, max_new_tokens, context_length, device):
    """
    指定したチェックポイントからGPTモデルをロードし、テキスト生成を行います。
    """
    # トークナイザの読み込み (gpt2のエンコーディング)
    tokenizer = tiktoken.get_encoding('gpt2')

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint['model_state_dict'].items()
    }

    # モデルの初期化（チェックポイントのconfigを利用）
    config = checkpoint['config']
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        context_length=context_length,
        tokenizer=tokenizer
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 入力プロンプトのエンコード
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # テキスト生成
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]

    # 生成結果が既に文字列の場合はそのまま出力、
    # トークン列の場合はデコードしてテキストに変換
    if isinstance(generated, str):
        output_text = generated
    else:
        # tensorの場合はリストに変換
        if isinstance(generated, torch.Tensor):
            generated = generated.tolist()
        output_text = tokenizer.decode(generated)

    print(output_text)

if __name__ == '__main__':
    generate_sample()
