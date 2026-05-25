from pathlib import Path
import torch
import tiktoken

# 1. 选择设备
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

print("使用设备:", device)

# 2. 找到当前脚本所在目录
project_dir = Path(__file__).parent
input_path = project_dir / "input.txt"

# 3. 如果没有 input.txt，就自动创建一个测试文本
if not input_path.exists():
    demo_text = """
Hello, world.
This is a tiny dataset for GPT training.
The model learns to predict the next token.
Hello, world.
This is a tiny dataset for GPT training.
The model learns to predict the next token.
""" * 100

    input_path.write_text(demo_text, encoding="utf-8")
    print("没有找到 input.txt，已自动创建一个测试版 input.txt")

# 4. 读取文本
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

print("\n原始文本前 100 个字符:")
print(text[:100])

# 5. 使用 GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 6. 文本 -> token ID
ids = enc.encode(text)

print("\n前 20 个 token ID:")
print(ids[:20])

print("\n前 20 个 token ID 解码回来:")
print(enc.decode(ids[:20]))

print("\n词表大小:", enc.n_vocab)
print("总 token 数:", len(ids))

# 7. 转成 PyTorch Tensor
data = torch.tensor(ids, dtype=torch.long)

# 8. 切分训练集和验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("\n训练 token 数:", len(train_data))
print("验证 token 数:", len(val_data))

# 9. 设置 batch 参数
block_size = 16
batch_size = 4

# 防止文本太短
if len(train_data) <= block_size + 1:
    raise ValueError("input.txt 太短，请放入更多文本，或者把 block_size 调小。")

# 10. 取 batch 的函数
def get_batch(split):
    d = train_data if split == "train" else val_data

    if len(d) <= block_size + 1:
        raise ValueError(f"{split} 数据太短，无法取 block_size={block_size} 的片段。")

    # 随机选 batch_size 个起点
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))

    # x 是输入
    x = torch.stack([
        d[i : i + block_size]
        for i in ix
    ])

    # y 是答案，比 x 整体右移一位
    y = torch.stack([
        d[i + 1 : i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# 11. 实际取一个 batch
x, y = get_batch("train")

print("\nx.shape:", x.shape)
print("y.shape:", y.shape)

print("\n第一条 x 解码:")
print(enc.decode(x[0].tolist()))

print("\n第一条 y 解码:")
print(enc.decode(y[0].tolist()))

print("\n验证右移关系：")
print("x[1:] 是否等于 y[:-1]:", torch.equal(x[0, 1:], y[0, :-1]))

# 12. 展示 x 和 y 的右移关系
print("\n逐 token 对照：模型看到 x，要预测 y")
for i in range(block_size):
    x_token = x[0, i].item()
    y_token = y[0, i].item()

    print(
        f"位置 {i:02d}: "
        f"x={x_token:<6} {repr(enc.decode([x_token])):<12} "
        f"--> y={y_token:<6} {repr(enc.decode([y_token]))}"
    )



