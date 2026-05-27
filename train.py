"""
train.py — 训练主循环
=====================
完整的训练流程：数据加载 → 模型创建 → 训练循环 → 评估 → 保存

训练循环流程图：

  ┌──────────────────────────────────────────────────┐
  │  for step in range(max_iters):                   │
  │    1. 取一个 batch (x, y)                        │
  │    2. 前向传播: model(x, y) → logits, loss       │
  │    3. 清空旧梯度: optimizer.zero_grad()           │
  │    4. 反向传播: loss.backward()                   │
  │    5. 梯度裁剪: clip_grad_norm_                   │
  │    6. 更新参数: optimizer.step()                  │
  │    7. 更新学习率: scheduler.step()                │
  │                                                  │
  │    每隔 eval_interval 步：                        │
  │      → 评估 train/val loss                       │
  │      → 生成样本文本                              │
  │      → 保存 checkpoint                           │
  └──────────────────────────────────────────────────┘

学习要点：
  训练的本质是 "算 loss → 求梯度 → 更新参数" 的循环。
  反向传播（backprop）自动帮你求出每个参数应该往哪个方向调整。
  优化器（AdamW）决定了调整的幅度和方式。
"""

import time
import torch

from config import (
    device, GPTConfig,
    batch_size, max_iters, eval_interval, eval_iters,
    learning_rate, weight_decay, grad_clip,
    gen_max_tokens, gen_temperature, gen_top_k,
    data_path, checkpoint_dir, best_model_path,
)
from dataset import load_and_tokenize, split_data, get_batch, decode, encode
from model import GPT
from utils import save_checkpoint, print_model_summary


def main():
    print("=" * 60)
    print("  🚀 GPT-2 训练开始")
    print("=" * 60)
    print(f"  设备: {device}")

    # ==============================================================
    # 1. 准备数据
    # ==============================================================
    print("\n📦 加载数据...")
    text, data = load_and_tokenize(data_path)
    train_data, val_data = split_data(data)

    print(f"  原始文本:    {len(text):,} 字符")
    print(f"  总 token:    {len(data):,}")
    print(f"  训练 token:  {len(train_data):,}")
    print(f"  验证 token:  {len(val_data):,}")

    # ==============================================================
    # 2. 创建模型
    # ==============================================================
    print("\n🔧 创建模型...")
    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    print_model_summary(model)

    # ==============================================================
    # 3. 创建优化器
    # ==============================================================
    # 学习要点：
    #   AdamW = Adam + 权重衰减（Weight Decay）
    #   Adam 维护每个参数的一阶动量(均值)和二阶动量(方差)，
    #   相比 SGD，Adam 对学习率不那么敏感，训练更稳定。
    #
    #   betas=(0.9, 0.95) 是 GPT-2 论文推荐的参数：
    #   - beta1=0.9: 一阶动量的衰减率（越大越"有惯性"）
    #   - beta2=0.95: 二阶动量的衰减率

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # ==============================================================
    # 4. 学习率调度器
    # ==============================================================
    # 学习要点：
    #   为什么不用固定学习率？
    #   → 训练初期，参数随机，大的 lr 可能导致不稳定
    #   → 训练中期，lr 最大，快速学习
    #   → 训练末期，lr 慢慢降低，精细调整（微调收敛）
    #
    #   CosineAnnealingLR: 余弦退火
    #   lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
    #   看起来像 cos 函数的上半部分：从 lr_max 平滑降到 lr_min

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=learning_rate * 0.1
    )

    # ==============================================================
    # 5. 评估函数
    # ==============================================================
    @torch.no_grad()
    def estimate_loss():
        """
        评估训练集和验证集的平均 loss

        学习要点：
          model.eval() 会关闭 Dropout（推理时不需要随机丢弃）
          model.train() 会重新开启 Dropout
          @torch.no_grad() 关闭梯度计算，省内存省时间
        """
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split, train_data, val_data, cfg.block_size, batch_size)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ==============================================================
    # 6. 生成样本文本
    # ==============================================================
    def generate_sample(prompt_text="", max_tokens=gen_max_tokens):
        """用当前模型生成一段文本，看看训练效果"""
        model.eval()

        if prompt_text:
            tokens = encode(prompt_text)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
        else:
            # 没有 prompt 就从一个随机 token 开始
            idx = torch.randint(0, cfg.vocab_size, (1, 1), device=device)

        generated = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=gen_temperature,
            top_k=gen_top_k,
        )

        result = decode(generated[0].tolist())
        model.train()
        return result

    # ==============================================================
    # 7. 训练循环 ⭐
    # ==============================================================
    print(f"\n🏋️ 开始训练: {max_iters} 步, batch_size={batch_size}, "
          f"block_size={cfg.block_size}")
    print("-" * 60)

    model.train()
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(max_iters):
        # --- 取一个 batch ---
        xb, yb = get_batch("train", train_data, val_data, cfg.block_size, batch_size)

        # --- 前向传播 ---
        logits, loss = model(xb, yb)

        # --- 反向传播 ---
        # 学习要点：
        #   zero_grad: 清空上一步的梯度（PyTorch 默认会累加梯度）
        #   set_to_none=True: 比 .zero_() 更高效（直接把梯度设为 None）
        #   backward(): 从 loss 出发，自动算出每个参数的梯度
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # --- 梯度裁剪 ---
        # 学习要点：
        #   如果梯度太大（梯度爆炸），参数更新幅度过大，训练会崩溃
        #   clip_grad_norm_ 会计算所有参数梯度的 L2 范数，
        #   如果超过 grad_clip，就等比缩小所有梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # --- 更新参数 ---
        optimizer.step()

        # --- 更新学习率 ---
        scheduler.step()

        # --- 定期评估和打印 ---
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]

            print(f"\n  📊 Step {step:5d}/{max_iters} | "
                  f"train loss: {losses['train']:.4f} | "
                  f"val loss: {losses['val']:.4f} | "
                  f"lr: {current_lr:.2e} | "
                  f"耗时: {elapsed:.1f}s")

            # 生成样本文本
            sample = generate_sample("莽莽苍苍")
            # 只显示前 120 个字符
            print(f"  📝 生成样本: {sample[:120]}...")

            # 保存最佳模型
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model, optimizer, step, losses["val"],
                    best_model_path,
                )

        # 每步打印简短日志
        elif step % 100 == 0:
            print(f"  step {step:5d} | loss: {loss.item():.4f}")

    # ==============================================================
    # 8. 训练完成
    # ==============================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  🎉 训练完成!")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
    print(f"  最佳 val loss: {best_val_loss:.4f}")
    print("=" * 60)

    # 最终保存
    final_path = f"{checkpoint_dir}/final_model.pt"
    save_checkpoint(model, optimizer, max_iters, best_val_loss, final_path)

    # 生成最终展示文本
    print("\n📝 最终生成展示：")
    for prompt in ["莽莽苍苍", "老瞎子", "小瞎子"]:
        result = generate_sample(prompt, max_tokens=150)
        print(f"\n  prompt: '{prompt}'")
        print(f"  生成: {result[:200]}")
        print()


if __name__ == "__main__":
    main()
