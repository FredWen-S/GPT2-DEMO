"""
generate.py — 文本生成脚本
===========================
加载训练好的模型，交互式生成文本。

用法：
  1. 默认交互模式：
     python generate.py

  2. 命令行指定参数：
     python generate.py --prompt "莽莽苍苍" --max_tokens 200 --temperature 0.8 --top_k 40

  3. 从指定 checkpoint 加载：
     python generate.py --checkpoint checkpoints/final_model.pt
"""

import argparse
import torch

from config import (
    device, GPTConfig,
    gen_temperature, gen_top_k, gen_max_tokens,
    best_model_path,
)
from model import GPT
from dataset import encode, decode
from utils import load_checkpoint


def load_model(checkpoint_path):
    """加载模型和权重"""
    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    step, loss = load_checkpoint(checkpoint_path, model)

    model.eval()
    print(f"  模型参数量: {model.count_parameters():,}")

    return model, cfg


def generate_text(model, cfg, prompt, max_tokens, temperature, top_k):
    """
    从 prompt 开始生成文本

    学习要点：
      整个生成过程：
      1. prompt 文本 → encode → token IDs
      2. token IDs → model.generate() → 新的 token IDs
      3. 新的 token IDs → decode → 生成的文本

      temperature 和 top_k 的交互效果：
      ┌────────────┬──────────┬──────────────────────┐
      │ temperature │  top_k   │       效果           │
      ├────────────┼──────────┼──────────────────────┤
      │    0.1     │    -     │ 几乎贪心，非常重复   │
      │    0.8     │   40     │ 连贯且有变化（推荐）  │
      │    1.0     │   100    │ 较为随机             │
      │    1.5     │   None   │ 非常随机，可能乱码   │
      └────────────┴──────────┴──────────────────────┘
    """
    if prompt:
        tokens = encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return decode(generated[0].tolist())


def interactive_mode(model, cfg):
    """交互模式：不断输入 prompt 不断生成"""
    print("\n" + "=" * 60)
    print("  🎮 交互式文本生成")
    print("  输入 prompt 后按回车生成，输入 'quit' 退出")
    print("  输入 't=0.5' 修改温度，输入 'k=50' 修改 top_k")
    print("  输入 'n=200' 修改生成长度")
    print("=" * 60)

    temperature = gen_temperature
    top_k = gen_top_k
    max_tokens = gen_max_tokens

    while True:
        print(f"\n  [温度={temperature} | top_k={top_k} | 长度={max_tokens}]")
        prompt = input("  📝 输入 prompt: ").strip()

        if prompt.lower() == "quit":
            print("  👋 再见!")
            break

        # 修改参数
        if prompt.startswith("t="):
            temperature = float(prompt[2:])
            print(f"  ✅ 温度已设为 {temperature}")
            continue
        if prompt.startswith("k="):
            top_k = int(prompt[2:])
            print(f"  ✅ top_k 已设为 {top_k}")
            continue
        if prompt.startswith("n="):
            max_tokens = int(prompt[2:])
            print(f"  ✅ 生成长度已设为 {max_tokens}")
            continue

        # 生成
        print("\n  ⏳ 生成中...")
        result = generate_text(model, cfg, prompt, max_tokens, temperature, top_k)
        print(f"\n  {'─' * 50}")
        print(f"  {result}")
        print(f"  {'─' * 50}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 文本生成")
    parser.add_argument("--checkpoint", type=str, default=best_model_path,
                        help="模型 checkpoint 路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="生成起始文本（不指定则进入交互模式）")
    parser.add_argument("--max_tokens", type=int, default=gen_max_tokens,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=gen_temperature,
                        help="温度参数")
    parser.add_argument("--top_k", type=int, default=gen_top_k,
                        help="Top-k 采样")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 GPT-2 文本生成器")
    print("=" * 60)
    print(f"  设备: {device}")
    print(f"  加载模型: {args.checkpoint}")

    model, cfg = load_model(args.checkpoint)

    if args.prompt is not None:
        # 命令行模式：直接生成
        result = generate_text(
            model, cfg, args.prompt,
            args.max_tokens, args.temperature, args.top_k,
        )
        print(f"\n  Prompt: {args.prompt}")
        print(f"  生成结果:\n")
        print(f"  {result}")
    else:
        # 交互模式
        interactive_mode(model, cfg)


if __name__ == "__main__":
    main()
