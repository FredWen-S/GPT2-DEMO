import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo2_1 import get_batch


class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0

        # 一次性算出 Q, K, V
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        # attention 输出投影
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # 因果 mask，下三角为 1
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # [B, T, C] -> [B, T, 3C] -> q/k/v 各 [B, T, C]
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention score: [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 未来位置设成 -inf，注意这里是 -inf，不是 inf
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # attention probability
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # [B, n_head, T, T] @ [B, n_head, T, head_dim]
        # -> [B, n_head, T, head_dim]
        y = att @ v

        # 拼回 [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)

        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.cfg.block_size, "输入长度 T 不能超过 block_size"

        # 位置编号：[0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # token embedding: [B, T] -> [B, T, C]
        tok_emb = self.transformer.wte(idx)

        # position embedding: [T] -> [T, C]
        pos_emb = self.transformer.wpe(pos)

        # 自动广播：[B, T, C] + [T, C] -> [B, T, C]
        x = self.transformer.drop(tok_emb + pos_emb)

        # 多个 Transformer Block
        for block in self.transformer.h:
            x = block(x)

        # 最终 LayerNorm
        x = self.transformer.ln_f(x)

        # 输出到词表 logits: [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    cfg = GPTConfig(
        vocab_size=50257,
        block_size=16,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )

    model = GPT(cfg).to(device)

    # 随机造一个 batch
    B = 2
    T = 8

    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(idx, targets)

    print("idx.shape:", idx.shape)
    print("targets.shape:", targets.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())

    num_params = sum(p.numel() for p in model.parameters())
    print("参数量:", num_params)

model = GPT(GPTConfig()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

max_iters = 5000
for step in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)  # 前向 + 算 loss
    optimizer.zero_grad(set_to_none=True)  # 清空旧梯度
    loss.backward()  # 反向传播求梯度
    optimizer.step()  # 更新权重

    if step % 500 == 0:
        print(f'step {step}: loss {loss.item():.4f}')


@torch.no_grad()
def estimate_loss(eval_iters=50):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 在训练循环里每隔一段调用：
# if step % 500 == 0:
#     l = estimate_loss()
#     print(f"step {step}: train {l['train']:.4f}, val {l['val']:.4f}")
# === 接在第 4 章模型代码之后 ===
text = open('input.txt').read()  # 任意文本；没有就用一段重复字符串测试
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

cfg = GPTConfig(vocab_size=len(chars), block_size=64,
                n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


def get_batch(bs=32):
    ix = torch.randint(len(data) - cfg.block_size - 1, (bs,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)


for step in range(3000):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True);
    loss.backward();
    opt.step()
    if step % 300 == 0: print(step, round(loss.item(), 4))

torch.save(model.state_dict(), 'my_gpt.pt')  # 保存权重
