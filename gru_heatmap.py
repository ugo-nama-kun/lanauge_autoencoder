# -*- coding: utf-8 -*-
"""
GRU版（提示コード準拠）で、
横軸: max_seq_len ∈ {1..20}
縦軸: image_types ∈ {1..10}
各条件の学習を1回ずつ走らせ、最終 total_loss を 2D ヒートマップ表示。

※ 学習の可視化や逐次プリントは抑制し、スイープに特化。
※ CUDA OOM 等が起きた条件は NaN として記録。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torchvision import transforms
import matplotlib.pyplot as plt

from mnist import MNISTHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ハイパーパラメータ（提示コード準拠）
# -------------------------------
input_dim = 28 * 28
base_vocab_size = 100
embedding_dim = 32
hidden_dim = 128
temperature = 1.0
batch_size = 128
max_steps = 10_000
lr = 3e-4
max_grad_norm = 1.0
w_entropy = 0.0

# -------------------------------
# Encoder（提示コード準拠）
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(vocab_size, hidden_dim, batch_first=True)

        self.initial_embedding = nn.Parameter(torch.randn(1, vocab_size), requires_grad=True)
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)  # 使わないが提示コード踏襲

        self.token_proj = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, x, tau):
        B = x.size(0)
        x = x.view(B, -1)

        context = F.elu(self.encoder(x))  # [B, H]
        h = context.unsqueeze(0)          # [1, B, H]

        # t=0 入力: 学習BOSベクトルを [B, 1, V] に拡張（GRUのinput_sizeは vocab_size）
        inp = self.initial_embedding.expand(B, 1, -1)  # [B, 1, V]

        tokens = []
        loss_entropy = 0.0
        for _ in range(self.max_seq_len):
            out, h = self.gru(inp, h)                  # out: [B, 1, H]
            logits = self.token_proj(out.squeeze(1))   # [B, V]

            y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # [B, V]
            tokens.append(y)

            probs = F.softmax(logits, dim=-1)
            loss_entropy = loss_entropy + dist.Categorical(probs=probs).entropy()

            # 次時刻の入力は logits（提示コード準拠）
            inp = logits.unsqueeze(1)                  # [B, 1, V]

        message = torch.stack(tokens, dim=1)           # [B, L, V]
        return message, (loss_entropy / self.max_seq_len).mean()

# -------------------------------
# Decoder（提示コード準拠）
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Linear(vocab_size, embedding_dim)

        self.initial_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, message):
        B = message.size(0)

        inp = self.initial_embedding.expand(B, -1, -1)  # [B, 1, E]
        h = torch.zeros(1, B, self.hidden_dim, device=message.device)  # [1, B, H]

        for t in range(self.max_seq_len):
            _, h = self.gru(inp, h)  # out は未使用、h 更新
            inp = self.embedding(message[:, t]).unsqueeze(1)  # 次時刻入力

        last_hidden = h.squeeze(0)       # [B, H]
        x_recon = self.fc(last_hidden)   # [B, 784]
        return x_recon

# -------------------------------
# 初期化（提示コード準拠）
# -------------------------------
def init_rnn_like_gru(m: nn.GRU):
    for name, param in m.named_parameters():
        if "weight_ih" in name:
            for w in param.chunk(3, dim=0):
                nn.init.xavier_uniform_(w)
        elif "weight_hh" in name:
            for w in param.chunk(3, dim=0):
                nn.init.orthogonal_(w)
        elif "bias" in name and param is not None:
            nn.init.zeros_(param)

def init_linear(m: nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def init_encoder_decoder(encoder: nn.Module, decoder: nn.Module):
    def _init(m):
        if isinstance(m, nn.Linear):
            init_linear(m)
        elif isinstance(m, nn.GRU):
            init_rnn_like_gru(m)
    encoder.apply(_init)
    decoder.apply(_init)

# -------------------------------
# データユーティリティ
# -------------------------------
@torch.no_grad()
def sample_batch(mnist_handler, image_types_val, batch_size, transform, device):
    xs = []
    for _ in range(batch_size):
        idx = np.random.randint(0, image_types_val)
        img = transform(mnist_handler.get_random_image(idx))
        xs.append(img)
    x = torch.stack(xs).to(device)  # [B, 1, 28, 28]
    return x

# -------------------------------
# 1条件を学習し、最終 total_loss を返す
# -------------------------------
def run_once(max_seq_len_val: int, image_types_val: int, seed: int = 123) -> float:
    """
    提示GRUモデルで、指定 max_seq_len / image_types を用いて学習を1回実行。
    返り値は学習の最終 step における total_loss（float）。
    例外や OOM の場合は NaN を返す。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # データ
    mnist_handler = MNISTHandler(train=True)
    transform = transforms.ToTensor()

    try:
        # モデル・最適化
        encoder = Encoder(input_dim, hidden_dim, max_seq_len_val, base_vocab_size).to(device)
        decoder = Decoder(base_vocab_size, embedding_dim, hidden_dim, max_seq_len_val, input_dim).to(device)
        init_encoder_decoder(encoder, decoder)

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        loss_fn = nn.BCELoss()

        final_total_loss = None
        for _ in range(max_steps):
            encoder.train(); decoder.train()
            optimizer.zero_grad(set_to_none=True)

            x = sample_batch(mnist_handler, image_types_val, batch_size, transform, device)
            message, loss_entropy = encoder(x, temperature)
            x_recon = decoder(message)

            loss = loss_fn(x_recon, x.view(x.size(0), -1)) - w_entropy * loss_entropy

            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm)
            optimizer.step()

            final_total_loss = float(loss.item())

        return final_total_loss

    except RuntimeError as e:
        # CUDA OOM 等は NaN 扱い
        if "out of memory" in str(e).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return float("nan")
    except Exception:
        return float("nan")

# -------------------------------
# スイープ & 可視化
# -------------------------------
def main():
    x_msl = list(range(1, 21))   # max_seq_len: 1..20（横軸）
    y_imgs = list(range(1, 11))  # image_types: 1..10（縦軸）

    loss_grid = np.zeros((len(y_imgs), len(x_msl)), dtype=np.float32)

    print("== sweep start ==")
    for iy, img_types in enumerate(y_imgs):
        for ix, msl in enumerate(x_msl):
            print(f"[image_types={img_types:2d}, max_seq_len={msl:2d}] ... ", end="", flush=True)
            loss_val = run_once(max_seq_len_val=msl, image_types_val=img_types, seed=123)
            loss_grid[iy, ix] = np.nan if np.isnan(loss_val) else loss_val
            print("OOM/NaN" if np.isnan(loss_val) else f"{loss_val:.4f}")

    # 2D ヒートマップ
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        extent=[min(x_msl)-0.5, max(x_msl)+0.5, min(y_imgs)-0.5, max(y_imgs)+0.5]
    )
    plt.colorbar(im, label="final total_loss")
    plt.xlabel("max_seq_len")
    plt.ylabel("image_types")
    plt.xticks(x_msl)
    plt.yticks(y_imgs)
    plt.title("Final total_loss over (max_seq_len × image_types) — GRU model")
    plt.tight_layout()
    plt.show()
    print("== sweep done ==")

if __name__ == "__main__":
    main()
