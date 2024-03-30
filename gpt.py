"""
Code following Andrej Karpathy's Let's Build GPT video
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
# --- Hyperparams ---
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 300
learning_rate = 1e-3
device = torch.device('mps') or ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 32
n_heads = 4
print(f"{device=}")
# -----------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# "Tokenizer". Simple; pros: small vocab size, cons: long sequences
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]
def decode(l): return [itos[i] for i in l]


encoded = encode("HALLO")
encoded, decode(encoded)

torch.manual_seed(1337)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y, = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x, masked=True):
        B, T, C = x.shape

        k, q = self.key(x), self.query(x)
        aff = q @ k.transpose(-2, -1) * C ** -.5 # Affinities
        if masked:
            aff = aff.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        aff = F.softmax(aff, dim=1)

        v = self.value(x)
        out = aff @ v
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.concat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.nnet = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.nnet(x)

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadedAttention(n_heads, n_embd//n_heads) # Get weights accounting fro affinities across many features (like grouped conoplution)
        self.fc = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd
        x = self.sa_heads(x) # (B, T, C)
        x = self.fc(x)
        logits = self.lm_head(x)
        # print(logits.shape)
        if targets is None:
            loss = None
        else:
            # Calculate loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits.shape, targets.shape)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            block = idx[:, -block_size:]
            logits, loss = self(block)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Bigram()
model = model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch()

    # Evaluate Loss
    logits, loss = model(xb, yb)
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(''.join(decode(model.generate(context, 500)[0].tolist())))
