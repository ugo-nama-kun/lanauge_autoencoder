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
max_seq_len = 5
base_vocab_size = 2
embedding_dim = 32
hidden_dim = 512
temperature = 0.5
batch_size = 128
max_steps = 10_000
num_plots = 100
lr = 3e-4
max_grad_norm = 1.0
dropout_p = 0.5
w_entropy = 0.0

# -------------------------------
# Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len, vocab_size, dropout_p):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # self.dropout = nn.Dropout(dropout_p)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru_ln = nn.LayerNorm(hidden_dim)

        self.initial_embedding = nn.Parameter(torch.randn(1, embedding_dim), requires_grad=True)
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.embed_ln = nn.LayerNorm(embedding_dim)

        self.token_proj = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, x, tau):
        B = x.size(0)
        x = x.view(B, -1)

        context = F.leaky_relu(self.encoder(x))  # [B, H]
        h = context.unsqueeze(0)  # [1, B, H]

        # t=0 の入力: 学習BOSベクトルを [B, 1, E] に拡張
        inp = self.embed_ln(self.initial_embedding).expand(B, 1, -1)  # [B, 1, E]

        tokens = []
        prev_msg = None
        loss_entropy = 0.0
        for t in range(self.max_seq_len):
            out, h = self.gru(inp, h)  # out: [B, 1, H]
            out = self.gru_ln(out.squeeze(1))  # [B, H]

            logits = self.token_proj(out)  # [B, V]
            probs = F.softmax(logits, dim=-1)
            y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # [B, V]
            tokens.append(y)

            # 次の時刻の入力を one-hot から埋め込みへ: y @ W で [B, E]
            emb_next = self.embed_ln(self.embedding(y))  # [B, E]
            inp = emb_next.unsqueeze(1)  # [B, 1, E]

            # エントロピー（平均は最後に取る）
            loss_entropy = loss_entropy + dist.Categorical(probs=probs).entropy()

        message = torch.stack(tokens, dim=1)  # [B, L, V]
        return message, (loss_entropy / self.max_seq_len).mean()


# -------------------------------
# Decoder
# -------------------------------
class GaussianDropout(nn.Module):
    # Gaussian dropout from : https://discuss.pytorch.org/t/gaussiandropout-implementation/151756/5

    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len, output_dim, dropout_p):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.dropout = GaussianDropout(dropout_p)
        self.embed_ln = nn.LayerNorm(embedding_dim)
        self.initial_embedding = nn.Parameter(torch.randn(1, embedding_dim), requires_grad=True)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru_ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, message):
        B = message.size(0)

        # z: [B, L, vocab_size], lengths: [B]
        inp = self.embed_ln(self.initial_embedding).expand(B, -1, -1)

        h = torch.zeros(1, B, self.hidden_dim, device=message.device)  # [1, B, H]
        for t in range(self.max_seq_len):
            out, h = self.gru(inp, h)  # out: [B, 1, H]
            out = self.gru_ln(out.squeeze(1))  # [B, H]

            inp = self.embedding(message[:, t]) # [B, L, emb]
            inp = self.embed_ln(self.dropout(inp)).unsqueeze(1)

        # 最後の有効時刻の hidden state を抽出
        last_hidden = h.squeeze(0)
        x_recon = torch.sigmoid(self.fc(self.gru_ln(self.dropout(last_hidden))))  # [B, output_dim]
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
encoder = Encoder(input_dim, hidden_dim, max_seq_len, base_vocab_size, dropout_p).to(device)
decoder = Decoder(base_vocab_size, embedding_dim, hidden_dim, max_seq_len, input_dim, dropout_p).to(device)
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
    message, loss_entropy = encoder(x, temperature)  # 長さ付き離散トークン系列
    x_recon = decoder(message)  # EOS以降を無視して復元

    loss = loss_fn(x_recon, x.view(x.size(0), -1)) - w_entropy * loss_entropy

    # print(message.shape)
    message_ids = message.argmax(dim=-1)  # [B, L]
    for index, msg in enumerate(message_ids):
        token_seq = msg.tolist()
        print(f"{index} : {token_seq}")
    # print(lengths)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm)

    optimizer.step()
    total_loss += loss.item()

    print(f"Epoch {step + 1}, Loss: {total_loss:.4f}")
    loss_hist.append(total_loss)
    plt.figure(0)
    plt.clf()
    plt.plot(loss_hist)
    plt.pause(0.01)

    # -------------------------------
    # 再構成可視化
    # -------------------------------
    if step % int(max_steps / num_plots) == 0:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            x = []
            for index in range(10):
                x.append(transform(mnist_handler.get_random_image(index)))
            x = torch.stack(x).to(device)

            message, _ = encoder(x, temperature)
            x_recon = decoder(message)
            x_recon = x_recon.view(-1, 1, 28, 28).cpu()

            message_ids = message.argmax(dim=-1)  # [B, L]
            for index in range(10):
                token_seq = message_ids[index].tolist()
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
