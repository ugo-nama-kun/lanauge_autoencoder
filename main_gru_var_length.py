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
# ハイパーパラメータ
# -------------------------------
input_dim = 28 * 28
max_seq_len = 30
base_vocab_size = 20
eos_token_id = base_vocab_size
vocab_size = base_vocab_size + 1  # EOSを追加

embedding_dim = 32
hidden_dim = 64
temperature = 1.0
batch_size = 128
max_steps = 10_000
num_plots = 100
lr = 3e-4
max_grad_norm = 1.0
w_entropy = 0.0

# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(vocab_size, hidden_dim, batch_first=True)

        self.initial_embedding = nn.Parameter(torch.randn(1, vocab_size), requires_grad=True)
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)

        self.token_proj = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.eos_token_id = eos_token_id

    def forward(self, x, tau):
        B = x.size(0)
        x = x.view(B, -1)

        context = F.elu(self.encoder(x))  # [B, H]
        h = context.unsqueeze(0)  # [1, B, H]

        # t=0 の入力: 学習BOSベクトルを [B, 1, E] に拡張
        inp = self.initial_embedding.expand(B, 1, -1)  # [B, 1, E]

        tokens = []
        loss_entropy = 0.0
        lengths = torch.full((B,), self.max_seq_len, dtype=torch.long, device=x.device)
        finished = torch.zeros(B, dtype=torch.bool, device=x.device)

        for t in range(self.max_seq_len):
            out, h = self.gru(inp, h)  # out: [B, 1, H]
            logits = self.token_proj(out.squeeze(1))

            y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # [B, V]
            tokens.append(y)

            # エントロピー（平均は最後に取る）
            probs = F.softmax(logits, dim=-1)
            loss_entropy = loss_entropy + dist.Categorical(probs=probs).entropy()

            inp = logits.unsqueeze(1)

            pred_id = y.argmax(dim=-1)
            newly_finished = (pred_id == self.eos_token_id) & (~finished)
            lengths[newly_finished] = t + 1
            finished |= newly_finished
            if finished.all():
                break


        message = torch.stack(tokens, dim=1)  # [B, L, V]
        return message, lengths, (loss_entropy / self.max_seq_len).mean()


# -------------------------------
# Decoder
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

    def forward(self, message, lengths):
        B = message.size(0)

        # z: [B, L, vocab_size], lengths: [B]
        inp = self.initial_embedding.expand(B, -1, -1)

        h = torch.zeros(1, B, self.hidden_dim, device=message.device)  # [1, B, H]
        h_list = []
        for t in range(self.max_seq_len):
            _, h = self.gru(inp, h)  # out: [B, 1, H]
            h_list.append(h)

            inp = self.embedding(message[:, t]).unsqueeze(1) # [B, L, emb]

        # 最後の有効時刻の hidden state を抽出
        last_hidden = torch.stack([h_list[l - 1][0, i] for i, l in enumerate(lengths)], dim=0)  # [B, H]

        last_hidden = last_hidden.squeeze(0)
        x_recon = self.fc((last_hidden))  # [B, output_dim]
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
encoder = Encoder(input_dim, hidden_dim, max_seq_len, vocab_size, eos_token_id).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, max_seq_len, input_dim).to(device)
init_encoder_decoder(encoder, decoder)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
loss_fn = nn.BCELoss()

# -------------------------------
# 学習ループ
# -------------------------------
print("start")
fig_visual, axs_visual = plt.subplots(2, 10, figsize=(15, 3))

loss_hist = []
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
    message, lengths, loss_entropy = encoder(x, temperature)  # 長さ付き離散トークン系列
    x_recon = decoder(message, lengths)  # EOS以降を無視して復元

    loss = loss_fn(x_recon, x.view(x.size(0), -1)) - w_entropy * loss_entropy

    # print(message.shape)
    message_ids = message.argmax(dim=-1)  # [B, L]
    for index, msg in enumerate(message_ids):
        token_seq = msg.tolist()
        # print(f"{index} : {token_seq}")
    # print(lengths)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm)

    optimizer.step()
    total_loss += loss.item()

    loss_hist.append(total_loss)
    plt.figure(0)
    plt.clf()
    plt.plot(loss_hist)
    plt.pause(0.01)

    # -------------------------------
    # 再構成可視化
    # -------------------------------
    if step % int(max_steps / num_plots) == 0:
        print(f"Epoch {step + 1}, Loss: {total_loss:.4f}")

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            x = []
            for index in range(10):
                x.append(transform(mnist_handler.get_random_image(index)))
            x = torch.stack(x).to(device)

            message, lengths, _ = encoder(x, temperature)
            x_recon = decoder(message, lengths)
            x_recon = x_recon.view(-1, 1, 28, 28).cpu()

            message_ids = message.argmax(dim=-1)  # [B, L]
            for index in range(10):
                length = lengths[index].item()
                token_seq = message_ids[index][:length].tolist()
                print(f"{index} : {token_seq}")

            for i in range(10):
                axs_visual[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
                axs_visual[1, i].imshow(x_recon[i].squeeze(), cmap='gray')
                axs_visual[0, i].axis('off')
                axs_visual[1, i].axis('off')
            plt.suptitle(f"Original (Top) and Reconstructed (Bottom) @ {step}")
        plt.pause(0.01)

plt.show()
print("finish")
