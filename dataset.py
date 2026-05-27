"""
dataset.py — 数据管线
=====================
文本 → 分词 → Tensor → 切分 → 取 batch

学习要点：
  整个数据流是 "字符串 → 整数序列 → 训练样本"
  1. tiktoken 把文本拆成 token，每个 token 对应一个整数 ID
  2. 切成 90% 训练 + 10% 验证
  3. get_batch 每次随机取一段连续的 token 序列作为训练样本
     x = [t0, t1, t2, ..., t_{n-1}]   （输入）
     y = [t1, t2, t3, ..., t_n]        （标签，右移一位）
     模型的任务就是：看到 x 中的每个位置，预测 y 中对应的下一个 token
"""

from pathlib import Path
import torch
import tiktoken

from config import device


# ============================================================
# 1. 初始化 GPT-2 分词器
# ============================================================
# 学习要点：
#   GPT-2 使用 BPE (Byte Pair Encoding) 分词
#   词表大小 50257（50000 个 BPE merge + 256 个字节 + 1 个特殊 token）
#   中文字符通常会被拆成多个 byte-level token

enc = tiktoken.get_encoding("gpt2")


# ============================================================
# 2. 数据加载与编码
# ============================================================
def load_and_tokenize(file_path):
    """
    读取文本文件 → tiktoken 编码 → 返回 (原始文本, token Tensor)

    参数:
      file_path: 文本文件路径

    返回:
      text: 原始文本字符串
      data: token ID 的 LongTensor, shape = [total_tokens]
    """
    path = Path(file_path)

    if not path.exists():
        # 如果文件不存在，创建一个小的测试数据
        demo_text = "Hello world. 这是一个测试文本。\n" * 200
        path.write_text(demo_text, encoding="utf-8")
        print(f"⚠️ 未找到 {file_path}，已自动创建测试文本")

    text = path.read_text(encoding="utf-8")
    ids = enc.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    return text, data


# ============================================================
# 3. 数据切分
# ============================================================
def split_data(data, train_ratio=0.9):
    """
    按比例切分训练集和验证集

    学习要点：
      为什么需要验证集？
      → 训练时模型在训练集上的 loss 会不断下降，但这不代表它真正"学会"了
      → 验证集是模型没见过的数据，用来检测是否"过拟合"
      → 如果 train_loss 很低但 val_loss 很高，说明模型在死记硬背

    参数:
      data:        完整的 token Tensor
      train_ratio: 训练集占比（默认 90%）

    返回:
      (train_data, val_data) 两个 Tensor
    """
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


# ============================================================
# 4. 取 Batch
# ============================================================
def get_batch(split, train_data, val_data, block_size, batch_size):
    """
    从数据中随机采样一个 batch

    学习要点：
      这是语言模型训练的核心！
      假设 block_size=4，文本 token 是 [a, b, c, d, e, f, g, h]
      随机选一个起点 i=2:
        x = [c, d, e, f]     输入
        y = [d, e, f, g]     标签（右移一位）

      模型需要学会：
        看到 [c]       → 预测 d
        看到 [c, d]    → 预测 e
        看到 [c, d, e] → 预测 f
        ...以此类推

      一个 batch 有 batch_size 条这样的序列，并行训练。

    参数:
      split:      "train" 或 "val"
      train_data: 训练集 Tensor
      val_data:   验证集 Tensor
      block_size: 序列长度（上下文窗口）
      batch_size: 一个 batch 有多少条序列

    返回:
      (x, y)  都是 shape [batch_size, block_size] 的 LongTensor
    """
    d = train_data if split == "train" else val_data

    # 随机选 batch_size 个起始位置
    ix = torch.randint(len(d) - block_size, (batch_size,))

    # 取出 x 和 y
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)


# ============================================================
# 5. 编解码辅助函数
# ============================================================
def decode(ids):
    """token ID 列表 → 文本字符串"""
    return enc.decode(ids)


def encode(text):
    """文本字符串 → token ID 列表"""
    return enc.encode(text)


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    from config import batch_size as bs

    print("=" * 50)
    print("  📦 数据管线测试")
    print("=" * 50)

    text, data = load_and_tokenize("input.txt")
    train_data, val_data = split_data(data)

    print(f"\n原始文本长度:  {len(text):,} 字符")
    print(f"总 token 数:   {len(data):,}")
    print(f"训练 token 数: {len(train_data):,}")
    print(f"验证 token 数: {len(val_data):,}")
    print(f"词表大小:      {enc.n_vocab:,}")

    block_size = 128
    x, y = get_batch("train", train_data, val_data, block_size, bs)

    print(f"\n--- batch 信息 ---")
    print(f"x.shape: {x.shape}  (batch_size={bs}, block_size={block_size})")
    print(f"y.shape: {y.shape}")
    print(f"设备: {x.device}")

    print(f"\n--- 第一条样本 ---")
    print(f"x 解码: {decode(x[0].tolist())[:80]}...")
    print(f"y 解码: {decode(y[0].tolist())[:80]}...")

    print(f"\n--- 验证右移关系 ---")
    print(f"x[0, 1:5] == y[0, 0:4] ? {torch.equal(x[0, 1:5], y[0, 0:4])}")

    print("\n✅ 数据管线测试通过！")