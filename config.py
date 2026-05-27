"""
config.py — 统一配置中心
========================
把所有超参数集中在一个地方，方便理解和调参。
"""

import torch

# ============================================================
# 1. 设备选择
# ============================================================
# 优先级：CUDA（NVIDIA GPU）> MPS（Mac GPU）> CPU
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# 2. 模型超参数
# ============================================================
class GPTConfig:
    """
    GPT-2 模型配置

    学习要点：
    - vocab_size: 词表大小，GPT-2 使用 BPE 分词，词表 50257 个 token
    - block_size: 上下文窗口长度（模型一次能"看到"多少个 token）
    - n_layer:    Transformer 层数（Block 的数量）
    - n_head:     多头注意力的头数
    - n_embd:     嵌入维度（必须能被 n_head 整除！因为每个头的维度 = n_embd / n_head）
    - dropout:    随机丢弃比率，防止过拟合
    """

    def __init__(
        self,
        vocab_size=50257,
        block_size=128,      # CPU 训练用小窗口（GPT-2 原版是 1024）
        n_layer=4,           # 4 层（GPT-2 原版是 12 层）
        n_head=4,            # 4 头（GPT-2 原版是 12 头）
        n_embd=128,          # 128 维（GPT-2 原版是 768 维）
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def __repr__(self):
        return (
            f"GPTConfig(vocab_size={self.vocab_size}, block_size={self.block_size}, "
            f"n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd}, "
            f"dropout={self.dropout})"
        )


# ============================================================
# 3. 训练超参数
# ============================================================
# 学习要点：
#   - batch_size 太大会内存不够，太小训练不稳定
#   - learning_rate 是最重要的超参数，3e-4 是常见的安全值
#   - weight_decay 相当于 L2 正则化，防止权重变得太大
#   - grad_clip 梯度裁剪，防止梯度爆炸（尤其在训练初期）

batch_size = 8               # CPU 用小 batch（有 GPU 可以开到 32-64）
max_iters = 3000             # 总训练步数（CPU 跑 3000 步大概几分钟）
eval_interval = 300          # 每隔多少步做一次评估
eval_iters = 20              # 评估时取多少个 batch 来平均 loss
learning_rate = 3e-4         # AdamW 初始学习率
weight_decay = 0.01          # 权重衰减
grad_clip = 1.0              # 梯度裁剪阈值（梯度的 L2 范数超过此值就缩放）

# ============================================================
# 4. 生成参数
# ============================================================
gen_max_tokens = 100         # 训练中途展示时生成的 token 数
gen_temperature = 0.8        # 生成温度（越低越确定，越高越随机）
gen_top_k = 40               # Top-k 采样（只从概率最高的 k 个中选）

# ============================================================
# 5. 文件路径
# ============================================================
data_path = "input.txt"
checkpoint_dir = "checkpoints"
best_model_path = "checkpoints/best_model.pt"


# ============================================================
# 运行此文件可查看配置
# ============================================================
if __name__ == "__main__":
    print(f"设备: {device}")
    print(f"模型配置: {GPTConfig()}")
    print(f"训练参数: batch_size={batch_size}, max_iters={max_iters}, lr={learning_rate}")
    print(f"生成参数: temperature={gen_temperature}, top_k={gen_top_k}")