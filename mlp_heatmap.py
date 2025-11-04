# -*- coding: utf-8 -*-
"""
max_seq_len ∈ {1..20} を横軸、image_types ∈ {1..10} を縦軸にして、
学習終了時（最終 step）の total_loss を 2D ヒートマップで可視化するスクリプト。
元コードの Encoder/Decoder/MNISTHandler/util.init_model をそのまま利用します。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from mnist import MNISTHandler
from util import init_model

# -------------------------------
# 固定ハイパラ（元コード準拠）
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 28 * 28
base_vocab_size = 2
eos_token_id = base_vocab_size
vocab_size = base_vocab_size
hidden_dim = 256
temperature = 1
batch_size = 64
max_steps = 2_000
lr = 1e-2

# -------------------------------
# 元コードのモデル定義
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, max_seq_len * vocab_size)
        )
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def forward(self, x, tau):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        logits = self.mlp(x)  # [B, L*V]
        logits = logits.view(batch_size, self.max_seq_len, self.vocab_size)  # [B, L, V]
        y = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, L, V]
        return y

class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = vocab_size * seq_len
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        z_flat = z.view(z.size(0), -1)  # [B, L*V]
        return torch.sigmoid(self.mlp(z_flat))  # [B, output_dim]

# -------------------------------
# 実験ユーティリティ
# -------------------------------
@torch.no_grad()
def _sample_batch(mnist_handler, image_types, batch_size, transform, device):
    xs = []
    for _ in range(batch_size):
        index = np.random.randint(0, image_types)
        img = transform(mnist_handler.get_random_image(index))
        xs.append(img)
    x = torch.stack(xs).to(device)  # [B, 1, 28, 28]
    return x

def run_once(max_seq_len_val: int, image_types_val: int, seed: int = 0) -> float:
    """指定の max_seq_len と image_types で元コード相当の学習を1回実行し、最終 total_loss を返す。"""
    # 乱数シード
    torch.manual_seed(seed)
    np.random.seed(seed)

    # データ
    mnist_handler = MNISTHandler(train=True)
    transform = transforms.ToTensor()

    # モデル・最適化
    encoder = init_model(Encoder(input_dim, hidden_dim, max_seq_len_val, vocab_size, eos_token_id).to(device))
    decoder = init_model(Decoder(vocab_size, max_seq_len_val, hidden_dim, input_dim).to(device))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.BCELoss()

    # 学習ループ（元コード準拠：各 step の total_loss は最後のバッチの loss）
    final_total_loss = None
    for step in range(max_steps):
        encoder.train(); decoder.train()
        optimizer.zero_grad()

        x = _sample_batch(mnist_handler, image_types_val, batch_size, transform, device)
        message = encoder(x, temperature)              # [B, L, V]
        x_recon = decoder(message)                     # [B, 28*28]
        loss = loss_fn(x_recon, x.view(x.size(0), -1))

        loss.backward()
        optimizer.step()

        final_total_loss = loss.item()

    return float(final_total_loss)

# -------------------------------
# スイープと可視化
# -------------------------------
def main():
    x_range = list(range(1, 21))   # max_seq_len: 1..20 (横軸)
    y_range = list(range(1, 11))  # image_types: 1..10 (縦軸)

    loss_grid = np.zeros((len(y_range), len(x_range)), dtype=np.float32)

    print("== sweep start ==")
    for iy, img_types in enumerate(y_range):
        for ix, msl in enumerate(x_range):
            print(f"[image_types={img_types:2d}, max_seq_len={msl:2d}] ...", end="", flush=True)
            loss_val = run_once(max_seq_len_val=msl, image_types_val=img_types, seed=123)
            loss_grid[iy, ix] = loss_val
            print(f" loss={loss_val:.4f}")

    # 可視化（ヒートマップ）
    plt.figure(figsize=(10, 6))
    im = plt.imshow(loss_grid, origin="lower", aspect="auto",
                    extent=[min(x_range)-0.5, max(x_range)+0.5, min(y_range)-0.5, max(y_range)+0.5])
    plt.colorbar(im, label="final total_loss")
    plt.xlabel("max_seq_len")
    plt.ylabel("image_types")

    # 目盛り（1刻み）
    plt.xticks(x_range)
    plt.yticks(y_range)
    plt.title("Final total_loss over (max_seq_len × image_types)")
    plt.tight_layout()
    plt.show()
    print("== sweep done ==")

if __name__ == "__main__":
    main()
