"""
model.py — GPT-2 模型定义
=========================
完整的 GPT-2 架构，包含：
  - CausalSelfAttention（因果自注意力）
  - MLP（前馈网络）
  - Block（Transformer 块）
  - GPT（完整模型）
  - generate()（自回归文本生成）

模型结构图（Pre-LN Transformer）：

  输入 token IDs [B, T]
       │
       ▼
  ┌─────────────┐
  │ Token Embed  │  wte: [vocab_size, n_embd]
  └──────┬──────┘
         │  +
  ┌──────┴──────┐
  │  Pos Embed   │  wpe: [block_size, n_embd]
  └──────┬──────┘
         │
         ▼
       Dropout
         │
         ▼
  ╔═══════════════════╗
  ║  Transformer Block ║  ×  n_layer
  ║  ┌───────────────┐ ║
  ║  │   LayerNorm   │ ║
  ║  │   Attention   │──── + 残差连接
  ║  │   LayerNorm   │ ║
  ║  │     MLP       │──── + 残差连接
  ║  └───────────────┘ ║
  ╚═══════════════════╝
         │
         ▼
     LayerNorm (ln_f)
         │
         ▼
  ┌─────────────┐
  │   lm_head    │  Linear: [n_embd, vocab_size]（与 wte 共享权重）
  └──────┬──────┘
         │
         ▼
  logits [B, T, vocab_size]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ============================================================
# 1. 因果自注意力
# ============================================================
class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制

    学习要点：
      "因果" = Causal = 每个位置只能看到自己和前面的 token，不能偷看未来
      实现方式：用一个下三角 mask 把未来位置的 attention score 设为 -inf

      "多头" = Multi-Head = 把 Q/K/V 拆成 n_head 份，每份独立做 attention
      好处：不同的头可以关注不同的模式（比如一个头关注语法，一个头关注语义）

      计算流程：
        1. 输入 x: [B, T, C] 通过线性层得到 Q, K, V
        2. 拆成 n_head 份
        3. Q @ K^T / sqrt(d_k) → attention score
        4. mask 掉未来位置
        5. softmax → attention weight
        6. weight @ V → 输出
        7. 拼回 [B, T, C] 通过输出投影
    """

    def __init__(self, cfg):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0, \
            f"n_embd ({cfg.n_embd}) 必须能被 n_head ({cfg.n_head}) 整除"

        # 一次性算出 Q, K, V（效率更高，等价于三个单独的 Linear）
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        # attention 输出投影
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # 因果 mask：下三角矩阵，1 表示可以看到，0 表示不能看到
        # 例如 block_size=4:
        #   [[1, 0, 0, 0],
        #    [1, 1, 0, 0],
        #    [1, 1, 1, 0],
        #    [1, 1, 1, 1]]
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, 序列长度, 嵌入维度

        # [B, T, C] → [B, T, 3C] → 拆成 Q, K, V 各 [B, T, C]
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 拆成多头: [B, T, C] → [B, n_head, T, head_dim]
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 计算 attention score: Q @ K^T / sqrt(d_k)
        # [B, n_head, T, head_dim] @ [B, n_head, head_dim, T] → [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 因果 mask：把未来位置设为 -inf（softmax 后变成 0）
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # softmax 得到 attention 权重
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 加权求和: [B, n_head, T, T] @ [B, n_head, T, head_dim] → [B, n_head, T, head_dim]
        y = att @ v

        # 拼回: [B, n_head, T, head_dim] → [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + dropout
        return self.resid_dropout(self.c_proj(y))


# ============================================================
# 2. 前馈网络 (MLP)
# ============================================================
class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network

    学习要点：
      结构很简单：Linear → GELU → Linear → Dropout
      第一个 Linear 把维度扩大 4 倍（n_embd → 4*n_embd），相当于"放大"特征
      第二个 Linear 压回来（4*n_embd → n_embd）
      GELU 是激活函数，比 ReLU 更平滑，GPT-2 原版就用 GELU
    """

    def __init__(self, cfg):
        super().__init__()

        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)        # [B, T, C] → [B, T, 4C]
        x = F.gelu(x)           # 激活函数
        x = self.c_proj(x)      # [B, T, 4C] → [B, T, C]
        x = self.dropout(x)
        return x


# ============================================================
# 3. Transformer Block
# ============================================================
class Block(nn.Module):
    """
    一个 Transformer Block = LayerNorm + Attention + 残差 + LayerNorm + MLP + 残差

    学习要点：
      Pre-LN vs Post-LN：
      - Post-LN（原始 Transformer）：x → Attention → Add → LN → MLP → Add → LN
      - Pre-LN（GPT-2 用的）：     x → LN → Attention → Add → LN → MLP → Add
      Pre-LN 训练更稳定，梯度流更通畅

      残差连接（Residual Connection）：
      输出 = 输入 + 子层(输入)
      好处：梯度可以直接"穿越"子层，解决深层网络的梯度消失问题
    """

    def __init__(self, cfg):
        super().__init__()

        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)

        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-LN + 残差
        x = x + self.attn(self.ln_1(x))   # 先 LN 再 Attention，然后加残差
        x = x + self.mlp(self.ln_2(x))    # 先 LN 再 MLP，然后加残差
        return x


# ============================================================
# 4. 完整 GPT 模型
# ============================================================
class GPT(nn.Module):
    """
    GPT-2 模型

    学习要点：
      权重共享（Weight Tying）：
      wte（token embedding）和 lm_head 共享同一组权重。
      直觉：embedding 把 token → 向量，lm_head 把向量 → token 概率，
      两个操作是"互逆"的，共享权重可以减少参数量且效果更好。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            # Token Embedding: 词表中每个 token 对应一个向量
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            # Position Embedding: 每个位置对应一个向量（学习得到的，不是 sincos）
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            # Embedding Dropout
            drop=nn.Dropout(cfg.dropout),
            # N 个 Transformer Block
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            # 最终的 LayerNorm
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))

        # 输出头：将隐藏状态投影到词表大小，得到每个 token 的概率
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享：wte 和 lm_head 用同一组权重
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-2 标准权重初始化

        学习要点：
          为什么不用默认初始化？
          → 默认的 Xavier/Kaiming 初始化是为 ReLU 设计的
          → GPT-2 论文用正态分布 N(0, 0.02)，训练更稳定
          → LayerNorm 的 gamma 初始化为 1，beta 初始化为 0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        前向传播

        参数:
          idx:     输入 token IDs, shape [B, T]
          targets: 目标 token IDs, shape [B, T]（训练时传入，推理时不传）

        返回:
          logits: 每个位置对词表的分数, shape [B, T, vocab_size]
          loss:   交叉熵损失（如果传了 targets 的话）
        """
        B, T = idx.size()

        assert T <= self.cfg.block_size, \
            f"输入长度 T={T} 超过了 block_size={self.cfg.block_size}"

        # 位置编号：[0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token Embedding + Position Embedding
        tok_emb = self.transformer.wte(idx)    # [B, T] → [B, T, C]
        pos_emb = self.transformer.wpe(pos)    # [T] → [T, C]

        # 相加（广播）+ Dropout
        x = self.transformer.drop(tok_emb + pos_emb)  # [B, T, C]

        # 通过 N 个 Transformer Block
        for block in self.transformer.h:
            x = block(x)

        # 最终 LayerNorm
        x = self.transformer.ln_f(x)

        # 输出到词表: [B, T, C] → [B, T, vocab_size]
        logits = self.lm_head(x)

        # 计算 loss（如果有 targets）
        loss = None
        if targets is not None:
            # cross_entropy 需要 [N, C] 和 [N] 的形状
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归文本生成

        学习要点：
          生成过程是一个循环：
          1. 把当前序列输入模型，拿到最后一个位置的 logits
          2. logits 除以 temperature：
             - T < 1: 概率分布变得更"尖"，模型更确定（接近贪心）
             - T > 1: 概率分布变得更"平"，模型更随机（更有创意）
             - T = 1: 原始概率
          3. Top-k 过滤：只保留概率最高的 k 个 token，其余设为 -inf
             → 避免采样到极其罕见的 token，生成更连贯
          4. softmax → 按概率采样一个 token
          5. 拼到序列末尾，重复

        参数:
          idx:            起始序列 [B, T]
          max_new_tokens: 生成多少个新 token
          temperature:    温度参数（默认 1.0）
          top_k:          Top-k 采样（默认 None = 不过滤）

        返回:
          完整序列 [B, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截取最后 block_size 个 token（位置编码有上限）
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                else idx[:, -self.cfg.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)

            # 只取最后一个位置的 logits: [B, vocab_size]
            logits = logits[:, -1, :]

            # 除以温度
            logits = logits / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 把低于第 k 大的值都设为 -inf
                logits[logits < v[:, [-1]]] = float("-inf")

            # softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            # 按概率采样一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # 拼到序列末尾
            idx = torch.cat([idx, idx_next], dim=1)  # [B, T+1]

        self.train()
        return idx

    def count_parameters(self):
        """统计总参数量"""
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 烟雾测试
# ============================================================
if __name__ == "__main__":
    from config import device as dev
    from utils import print_model_summary

    print("=" * 50)
    print("  🔧 模型烟雾测试")
    print("=" * 50)

    cfg = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )
    print(f"\n配置: {cfg}")
    print(f"设备: {dev}")

    model = GPT(cfg).to(dev)

    # 打印模型结构
    print_model_summary(model)

    # 测试前向传播
    B, T = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

    logits, loss = model(idx, targets)

    print(f"\n--- 前向传播测试 ---")
    print(f"输入 shape:  {idx.shape}")
    print(f"输出 shape:  {logits.shape}")
    print(f"Loss:        {loss.item():.4f}")
    print(f"期望 loss ≈ -ln(1/50257) ≈ {math.log(50257):.4f}")

    # 测试生成
    start = torch.zeros((1, 1), dtype=torch.long, device=dev)
    generated = model.generate(start, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"\n--- 生成测试（随机权重，输出是乱码，这是正常的）---")
    print(f"生成 shape: {generated.shape}")

    print("\n✅ 模型烟雾测试通过！")