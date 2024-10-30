# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import os
import torch
import urllib.request

# Import from local files
from gpt_model import GPTModel, create_dataloader_v1, generate_text_simple
from gpt_train import text_to_token_ids, token_ids_to_text


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=52, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)

                # 移除初始输入的词汇
        output_text = decoded_text.replace(start_context, "")

        # 确保移除后文本的开头没有多余的空格
        output_text = output_text.strip()
        print(output_text.replace("\n", " "))  # Compact print format


def main(gpt_config, hparams):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "novel.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()


    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "ctx_len": 1024,       # Shortened context length (orig: 1024)
        "emb_dim": 768,       # Embedding dimension
        "n_heads": 12,        # Number of attention heads
        "n_layers": 12,       # Number of layers
        "drop_rate": 0.1,     # Dropout rate
        "qkv_bias": False     # Query-key-value bias
    }
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load("model.pth",weights_only=True))
    model.to(device)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=hparams["batch_size"],
        max_length=gpt_config["ctx_len"],
        stride=gpt_config["ctx_len"],
        drop_last=True,
        shuffle=True
    )

    start_context="Where"

    generate_and_print_sample(model,train_loader.dataset.tokenizer,device,start_context)

if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "ctx_len": 256,       # Shortened context length (orig: 1024)
        "emb_dim": 768,       # Embedding dimension
        "n_heads": 12,        # Number of attention heads
        "n_layers": 12,       # Number of layers
        "drop_rate": 0.1,     # Dropout rate
        "qkv_bias": False     # Query-key-value bias
    }

    OTHER_HPARAMS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    main(GPT_CONFIG_124M, OTHER_HPARAMS)


