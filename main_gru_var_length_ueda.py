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
max_seq_len = 20
base_vocab_size = 8
eos_token_id = base_vocab_size
vocab_size = base_vocab_size + 1  # EOSを追加
embedding_dim = 256
hidden_dim = 256
temperature = 0.5
batch_size = 256
max_steps = 1_000
lr = 1e-5
max_grad_norm = 10
dropout_p = 0.3

# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id, dropout_p):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(vocab_size, hidden_dim, batch_first=True)
        self.token_proj = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def forward(self, x, tau):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        context = F.gelu(self.fc(x))  # [B, H]
        context = self.dropout(context)
        h = context.unsqueeze(0)  # [1, B, H]

        input_token = torch.zeros(batch_size, 1, self.vocab_size, device=x.device)
        input_token[:, :, 0] = 1.0  # BOS

        tokens = []
        lengths = torch.full((batch_size,), self.max_seq_len, dtype=torch.long, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for t in range(self.max_seq_len):
            out, h = self.gru(input_token, h)  # [B, 1, H]
            out = self.dropout(out)
            logits = self.token_proj(out.squeeze(1))  # [B, vocab_size]
            y = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, vocab_size]
            tokens.append(y)
            input_token = y.unsqueeze(1)

            pred_id = y.argmax(dim=-1)
            newly_finished = (pred_id == self.eos_token_id) & (~finished)
            lengths[newly_finished] = t + 1
            finished |= newly_finished
            if finished.all():
                break

        message = torch.stack(tokens, dim=1)  # [B, ≤max_seq_len, vocab_size]
        return message, lengths


# -------------------------------
# Decoder
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_p):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, message, lengths):
        # z: [B, L, vocab_size], lengths: [B]
        z_emb = self.embedding(message)  # [B, L, emb]
        z_emb = self.dropout(z_emb)
        packed = nn.utils.rnn.pack_padded_sequence(z_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [B, L, H]

        # 最後の有効時刻の hidden state を抽出
        last_hidden = torch.stack([out[i, l - 1] for i, l in enumerate(lengths)], dim=0)  # [B, H]
        x_recon = torch.sigmoid(self.fc(last_hidden))  # [B, output_dim]
        return x_recon

# -------------------------------
# 初期化
# -------------------------------
def init_rnn_like_gru(m: nn.GRU):
    # 単層・多層 / 双方向にも対応
    for name, param in m.named_parameters():
        if "weight_ih" in name:
            # input->hidden を (r, z, n) の3ブロックに分割してXavier
            for w in param.chunk(3, dim=0):
                nn.init.xavier_uniform_(w)
        elif "weight_hh" in name:
            # hidden->hidden も (r, z, n) で直交初期化
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
# データセット（MNIST）
# -------------------------------

mnist_handler = MNISTHandler(train=True)
transform = transforms.Compose([
    transforms.ToTensor()
])

# -------------------------------
# モデルと最適化
# -------------------------------
encoder = Encoder(input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id, dropout_p).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, input_dim, dropout_p).to(device)
init_encoder_decoder(encoder, decoder)

optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=5e-4)
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
    # print(x.shape)

    optimizer.zero_grad()
    message, lengths = encoder(x, temperature)  # 長さ付き離散トークン系列
    x_recon = decoder(message, lengths)  # EOS以降を無視して復元

    loss = loss_fn(x_recon, x.view(x.size(0), -1))

    # print(message.shape)
    message_ids = message.argmax(dim=-1)  # [B, L]
    for index, msg in enumerate(message_ids):
        length = lengths[index].item()
        token_seq = msg[:length].tolist()
        print(f"{index} : {token_seq}")
    # print(lengths)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm)

    optimizer.step()
    total_loss += loss.item()

    print(f"Epoch {step + 1}, Loss: {total_loss:.4f}")
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

    message_ids = message.argmax(dim=-1)  # [B, L]
    for index in range(10):
        length = lengths[index].item()
        token_seq = message_ids[index][:length].tolist()
        print(f"{index} : {token_seq}")

    plt.figure()
    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axs[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
        axs[1, i].imshow(x_recon[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.suptitle("Original (Top) and Reconstructed (Bottom)")

plt.show()
print("finish")
