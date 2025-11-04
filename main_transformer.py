import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from mnist import MNISTHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ハイパーパラメータ
# -------------------------------
input_dim = 28 * 28
max_seq_len = 10
base_vocab_size = 100
eos_token_id = base_vocab_size
vocab_size = base_vocab_size + 1  # EOSトークン追加
embedding_dim = 64
hidden_dim = 64
nhead = 4
nlayers = 2
temperature = 0.5
batch_size = 64
max_steps = 10_000
lr = 1e-4

# -------------------------------
# Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# -------------------------------
# Transformer Encoder
# -------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id, nhead, nlayers):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.token_proj = nn.Linear(hidden_dim, vocab_size)

        self.token_input = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x, tau):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h0 = F.relu(self.fc(x))  # [B, H]

        # トークン列を展開
        token_input = self.token_input.expand(batch_size, -1, -1).clone()  # [B, L, H]

        # 先頭位置に h0 を挿入（置換）
        token_input[:, 0, :] = h0  # [B, L, H]

        # 位置エンコーディング付与 → Transformer
        token_input = self.pos_enc(token_input)  # [B, L, H]
        out = self.transformer(token_input)  # [B, L, H]

        # 語彙へ射影 → Gumbel-Softmax サンプリング
        logits = self.token_proj(out)  # [B, L, V]
        y = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, L, V]

        # EOS による長さ計算
        token_ids = y.argmax(dim=-1)  # [B, L]
        finished = token_ids == self.eos_token_id
        lengths = finished.float().argmax(dim=1) + 1
        lengths[~finished.any(dim=1)] = self.max_seq_len

        return y, lengths

# -------------------------------
# Transformer Decoder
# -------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, nhead, nlayers):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=512)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, z, lengths):
        z_emb = self.embedding(z)  # [B, L, D]
        z_emb = self.pos_enc(z_emb)
        packed = nn.utils.rnn.pack_padded_sequence(z_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        out = self.transformer(unpacked)
        last_hidden = torch.stack([out[i, l-1] for i, l in enumerate(lengths)], dim=0)
        x_recon = torch.sigmoid(self.fc(last_hidden))  # [B, output_dim]
        return x_recon

# -------------------------------
# データセット（MNIST）
# -------------------------------
mnist_handler = MNISTHandler(train=True)
transform = transforms.Compose([transforms.ToTensor()])

# -------------------------------
# モデルと最適化
# -------------------------------
encoder = TransformerEncoder(input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id, nhead, nlayers).to(device)
decoder = TransformerDecoder(vocab_size, embedding_dim, hidden_dim, input_dim, nhead, nlayers).to(device)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-5)
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

    x = []
    for _ in range(batch_size):
        index = np.random.randint(0, 10)
        x.append(transform(mnist_handler.get_random_image(index)))
    x = torch.stack(x).to(device)

    optimizer.zero_grad()
    message, lengths = encoder(x, temperature)
    x_recon = decoder(message, lengths)
    loss = loss_fn(x_recon, x.view(x.size(0), -1))

    message_ids = message.argmax(dim=-1)
    for index, msg in enumerate(message_ids[:5]):
        token_seq = msg[:lengths[index]].tolist()
        print(f"{index} : {token_seq}")

    loss.backward()
    optimizer.step()
    total_loss += loss.item()
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
with torch.no_grad():
    x = []
    for index in range(10):
        x.append(transform(mnist_handler.get_random_image(index)))
    x = torch.stack(x).to(device)

    message, lengths = encoder(x, temperature)
    x_recon = decoder(message, lengths)
    x_recon = x_recon.view(-1, 1, 28, 28).cpu()
    message_ids = message.argmax(dim=-1)

    for index in range(10):
        token_seq = message_ids[index][:lengths[index]].tolist()
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
