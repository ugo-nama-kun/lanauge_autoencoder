# -*- coding: utf-8 -*-
"""
横軸：vocab_size の指数 k ∈ {1..20}（vocab_size = 2**k）
縦軸：image_types ∈ {1..10}
各条件で学習を 1 回走らせ、最終 step の total_loss を 2D ヒートマップで可視化するスクリプト。
※ メモリ不足(OOM)などで失敗した条件は NaN として記録します。

前提：
- MNISTHandler, init_model は提示コードと同一のパスで import 可能
- 使用モデルは提示の one-hot（系列長1）版 Encoder/Decoder を流用
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
# 固定ハイパラ（提示コード準拠）
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 28 * 28
hidden_dim = 256
temperature = 1
batch_size = 64
max_steps = 2_000
lr = 1e-2

# -------------------------------
# 提示コードのモデル定義（one-hot 表現）
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.vocab_size = vocab_size

    def forward(self, x, tau):
        b = x.size(0)
        x = x.view(b, -1)            # [B, 784]
        logits = self.mlp(x)         # [B, V]
        y = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, V] (one-hot)
        return y

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        z_flat = z.view(z.size(0), -1)  # [B, V]
        return torch.sigmoid(self.mlp(z_flat))  # [B, 784]

# -------------------------------
# データユーティリティ
# -------------------------------
@torch.no_grad()
def sample_batch(mnist_handler, image_types, batch_size, transform, device):
    xs = []
    for _ in range(batch_size):
        idx = np.random.randint(0, image_types)
        img = transform(mnist_handler.get_random_image(idx))
        xs.append(img)
    x = torch.stack(xs).to(device)  # [B, 1, 28, 28]
    return x

# -------------------------------
# 1 条件の学習を実行し、最終 total_loss を返す
# -------------------------------
def run_once(vocab_exp: int, image_types_val: int, seed: int = 123) -> float:
    """
    vocab_exp: vocab_size = 2**vocab_exp
    返り値: 学習最終 step の total_loss（float）
    例外や OOM の場合は np.nan を返す
    """
    vocab_size = 2 ** vocab_exp

    # 乱数シード
    torch.manual_seed(seed)
    np.random.seed(seed)

    # データ
    mnist_handler = MNISTHandler(train=True)
    transform = transforms.ToTensor()

    try:
        # モデル・最適化
        encoder = init_model(Encoder(input_dim, hidden_dim, vocab_size).to(device))
        decoder = init_model(Decoder(vocab_size, hidden_dim, input_dim).to(device))
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=lr
        )
        loss_fn = nn.BCELoss()

        final_total_loss = None
        for step in range(max_steps):
            encoder.train(); decoder.train()
            optimizer.zero_grad(set_to_none=True)

            x = sample_batch(mnist_handler, image_types_val, batch_size, transform, device)
            message = encoder(x, temperature)            # [B, V]
            x_recon = decoder(message)                   # [B, 784]
            loss = loss_fn(x_recon, x.view(x.size(0), -1))

            loss.backward()
            optimizer.step()

            final_total_loss = float(loss.item())

        return final_total_loss

    except RuntimeError as e:
        # CUDA OOM 等は NaN で扱う
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        return float("nan")
    except Exception:
        return float("nan")

# -------------------------------
# スイープ & 可視化
# -------------------------------
def main():
    x_exponents = list(range(1, 14))   # vocab_size = 2**k, k=1..13（横軸は指数）
    y_imgtypes = list(range(1, 11))    # image_types = 1..10（縦軸）

    loss_grid = np.zeros((len(y_imgtypes), len(x_exponents)), dtype=np.float32)

    print("== sweep start ==")
    for iy, img_types in enumerate(y_imgtypes):
        for ix, k in enumerate(x_exponents):
            print(f"[image_types={img_types:2d}, vocab_exp(k)={k:2d} -> V=2**{k:2d}={2**k:,}] ... ", end="", flush=True)
            loss_val = run_once(vocab_exp=k, image_types_val=img_types, seed=123)
            loss_grid[iy, ix] = np.nan if np.isnan(loss_val) else loss_val
            status = "OOM/NaN" if np.isnan(loss_val) else f"{loss_val:.4f}"
            print(status)

    # 可視化（ヒートマップ）
    plt.figure(figsize=(11, 6))
    im = plt.imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        extent=[min(x_exponents)-0.5, max(x_exponents)+0.5, min(y_imgtypes)-0.5, max(y_imgtypes)+0.5]
    )
    plt.colorbar(im, label="final total_loss")
    plt.xlabel("vocab_size exponent k (vocab_size = 2^k)")
    plt.ylabel("image_types")

    # 目盛り
    plt.xticks(x_exponents)
    plt.yticks(y_imgtypes)

    # 参考：上部に実際の vocab_size（2^k）を刻むセカンダリ軸（大きすぎる場合は省略可）
    ax = plt.gca()
    secax = ax.secondary_xaxis('top', functions=(lambda k: k, lambda k: k))
    secax.set_xticks(x_exponents)
    secax.set_xticklabels([f"{2**k:,}" for k in x_exponents], rotation=45, ha="left")
    secax.set_xlabel("vocab_size (= 2^k)")

    plt.title("Final total_loss over (vocab_size exponent × image_types)")
    plt.tight_layout()
    plt.show()
    print("== sweep done ==")

if __name__ == "__main__":
    main()
