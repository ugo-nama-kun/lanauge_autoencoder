import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from mnist import MNISTHandler
from util import init_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ハイパーパラメータ
# -------------------------------
input_dim = 28 * 28
max_seq_len = 10
base_vocab_size = 20
eos_token_id = base_vocab_size
vocab_size = base_vocab_size
hidden_dim = 256
temperature = 1
batch_size = 256
max_steps = 1_000
lr = 1e-2

# -------------------------------
# Encoder（MLPベース）
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
        x = x.view(batch_size, -1)  # [B, input_dim]
        logits = self.mlp(x)  # [B, max_seq_len * vocab_size]
        logits = logits.view(batch_size, self.max_seq_len, self.vocab_size)  # [B, L, V]
        y = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, L, V]
        return y

# -------------------------------
# Decoder（MLPベース）
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = vocab_size * seq_len  # flatten用
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        z_flat = z.view(z.size(0), -1)  # [B, L*V]
        return F.sigmoid(self.mlp(z_flat))        # [B, output_dim]

# -------------------------------
# データセット（MNIST）
# -------------------------------
mnist_handler = MNISTHandler(train=True)
transform = transforms.ToTensor()

# -------------------------------
# モデルと最適化
# -------------------------------
encoder = init_model(Encoder(input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id).to(device))
decoder = init_model(Decoder(vocab_size, max_seq_len, hidden_dim, input_dim).to(device))
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
loss_fn = nn.BCELoss()

# -------------------------------
# 学習ループ
# -------------------------------
print("start")
loss_hist = []
plt.figure()

for step in range(max_steps):
    encoder.train()
    decoder.train()
    total_loss = 0

    # 入力画像の取得
    x = []
    for _ in range(batch_size):
        index = np.random.randint(0, 10)
        img = transform(mnist_handler.get_random_image(index))
        x.append(img)
    x = torch.stack(x).to(device)  # [B, 1, 28, 28]

    optimizer.zero_grad()
    message = encoder(x, temperature)       # [B, L, V], [B]
    x_recon = decoder(message)                       # [B, 28*28]
    loss = loss_fn(x_recon, x.view(x.size(0), -1))

    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    if step % 20 == 0:
        print(f"Step {step + 1}, Loss: {total_loss:.4f}")
        loss_hist.append(total_loss)
        plt.clf()
        plt.plot(loss_hist)
        plt.pause(0.01)

# -------------------------------
# 再構成可視化
# -------------------------------
encoder.eval()
decoder.eval()
plt.figure()
with torch.no_grad():
    x = []
    for index in range(10):
        x.append(transform(mnist_handler.get_random_image(index)))
    x = torch.stack(x).to(device)

    message = encoder(x, temperature)
    x_recon = decoder(message)
    x_recon = x_recon.view(-1, 1, 28, 28).cpu()

    message_ids = message.argmax(dim=-1)  # [B, L]
    for index in range(10):
        token_seq = message_ids[index].tolist()
        print(f"{index} : {token_seq}")

    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axs[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
        axs[1, i].imshow(x_recon[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.suptitle("Original (Top) and Reconstructed (Bottom)")
plt.show()

print("finish")
