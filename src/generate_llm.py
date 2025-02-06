import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from util.llm import GPT

tokenizer = tiktoken.get_encoding('gpt2')
context_length = 512

def generate_sample():
      # number of tokens processed in a single batch

    load_path = "./models/gpt_model_epoch_16000.pth"  # 保存時のファイル名に合わせる
    checkpoint = torch.load(load_path, map_location='cuda')
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model_state_dict'].items()}

    loaded_model = GPT(
        vocab_size=checkpoint['config']['vocab_size'],
        d_model=checkpoint['config']['d_model'],
        n_heads=checkpoint['config']['n_heads'],
        n_layers=checkpoint['config']['n_layers'],
        context_length=context_length,
        tokenizer=tokenizer
        ).to('cuda')

    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    with torch.no_grad():
        input = torch.tensor(tokenizer.encode("I'm praying: "), dtype=torch.long, device='cuda').unsqueeze(0)
        print(loaded_model.generate(input, max_new_tokens=500)[0])

if __name__ == '__main__':
    generate_sample()
