"""
utils.py — 工具函数
===================
模型保存/加载 + 模型结构打印。
"""

import os
import torch


def save_checkpoint(model, optimizer, step, loss, path):
    """
    保存训练检查点

    学习要点：
      为什么要保存优化器状态？
      → AdamW 内部为每个参数维护了一阶动量(m)和二阶动量(v)，
        如果不保存，续训时这些状态从零开始，效果会变差。

      state_dict() vs 保存整个模型对象：
      → state_dict() 只保存参数字典，更灵活、跨版本兼容。
      → 直接 torch.save(model) 会绑定类定义，换了代码就可能加载失败。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

    print(f"  💾 checkpoint 已保存: {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    加载训练检查点，支持断点续训

    参数:
      path:      checkpoint 文件路径
      model:     已创建的模型（会把权重加载进去）
      optimizer: 可选，传入则同时恢复优化器状态

    返回:
      (step, loss)  上次保存时的训练步数和 loss
    """
    if not os.path.exists(path):
        print(f"  ⚠️ 找不到 checkpoint: {path}")
        return 0, None

    # map_location='cpu' 确保在没有 GPU 的机器上也能加载
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✅ 模型权重已加载: {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"  ✅ 优化器状态已加载")

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", None)
    print(f"  📍 从第 {step} 步继续，上次 loss: {loss}")

    return step, loss


def print_model_summary(model):
    """
    打印模型结构和各层参数量

    学习要点：
      GPT-2 的参数主要集中在两个地方：
      1. Embedding 层（vocab_size × n_embd，词表大就很大）
      2. Transformer Block 里的 Linear 层

      由于 wte（token embedding）和 lm_head 共享权重，
      实际参数量 = 总参数量 - 一份 embedding 的大小
    """
    print("\n" + "=" * 70)
    print("  📐 模型结构总览")
    print("=" * 70)

    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        # 格式：参数名 | 形状 | 参数数量
        print(f"  {name:45s} {str(list(param.shape)):20s} {count:>10,}")

    print("-" * 70)
    print(f"  {'总参数量':42s} {'':20s} {total:>10,}")
    print(f"  约 {total / 1e6:.2f}M 参数")
    print("=" * 70)

    return total


if __name__ == "__main__":
    print("utils.py 工具模块加载成功！")
    print("提供函数: save_checkpoint, load_checkpoint, print_model_summary")
