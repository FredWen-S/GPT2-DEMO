"""
load_pretrained.py — 加载 HuggingFace GPT-2 预训练权重
======================================================
把 OpenAI 官方训练好的 GPT-2 权重加载到你自己写的模型中！

学习要点：
  这是验证你写的模型架构是否正确的终极测试：
  如果你的模型结构和 GPT-2 一致，就能完美加载官方权重。
  加载后不需要训练，就能直接生成高质量的英文文本！

  HuggingFace 的 GPT-2 和你写的模型有一个关键区别：
  - HuggingFace 用 Conv1D（实际是转置的 Linear）
  - 你的模型用的是标准 Linear
  所以加载时需要把 Conv1D 的权重做一次转置

用法：
  python load_pretrained.py

注意：
  首次运行需要下载模型（约 500MB），请确保网络通畅。
  如果下载慢，可以设置 HuggingFace 镜像：
    set HF_ENDPOINT=https://hf-mirror.com
"""

import torch
from model import GPT
from config import device
from dataset import decode


def load_gpt2_from_huggingface(model_name="gpt2"):
    """
    从 HuggingFace 加载 GPT-2 预训练权重

    参数:
      model_name: 模型名称
        - "gpt2":        124M 参数（推荐，CPU 能跑）
        - "gpt2-medium": 350M 参数
        - "gpt2-large":  774M 参数
        - "gpt2-xl":     1558M 参数

    返回:
      你自己写的 GPT 模型（已加载预训练权重）
    """
    from transformers import GPT2LMHeadModel

    print(f"\n📥 从 HuggingFace 下载 {model_name}...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()

    # ----------------------------------------------------------
    # 1. 根据 HuggingFace 模型的配置，创建你自己的模型
    # ----------------------------------------------------------
    hf_config = hf_model.config

    # GPT-2 各版本的配置对照：
    #   gpt2:        n_layer=12, n_head=12, n_embd=768   (124M)
    #   gpt2-medium: n_layer=24, n_head=16, n_embd=1024  (350M)
    #   gpt2-large:  n_layer=36, n_head=20, n_embd=1280  (774M)
    #   gpt2-xl:     n_layer=48, n_head=25, n_embd=1600  (1558M)

    from config import GPTConfig
    cfg = GPTConfig(
        vocab_size=hf_config.vocab_size,      # 50257
        block_size=hf_config.n_positions,     # 1024
        n_layer=hf_config.n_layer,            # 12
        n_head=hf_config.n_head,              # 12
        n_embd=hf_config.n_embd,              # 768
        dropout=hf_config.resid_pdrop,        # 0.1
    )

    print(f"  配置: {cfg}")

    model = GPT(cfg)

    # ----------------------------------------------------------
    # 2. 权重映射
    # ----------------------------------------------------------
    # 学习要点：
    #   HuggingFace 的 GPT2 模型的 key 和你写的模型的 key 略有不同：
    #
    #   HuggingFace key                         你的 key
    #   ─────────────────────────────────────    ────────────────────────────────
    #   transformer.wte.weight                   transformer.wte.weight           ✅ 直接对应
    #   transformer.wpe.weight                   transformer.wpe.weight           ✅ 直接对应
    #   transformer.h.0.ln_1.weight              transformer.h.0.ln_1.weight      ✅ 直接对应
    #   transformer.h.0.attn.c_attn.weight       transformer.h.0.attn.c_attn.weight  ⚠️ 需要转置
    #   transformer.h.0.attn.c_proj.weight       transformer.h.0.attn.c_proj.weight  ⚠️ 需要转置
    #   transformer.h.0.mlp.c_fc.weight          transformer.h.0.mlp.c_fc.weight     ⚠️ 需要转置
    #   transformer.h.0.mlp.c_proj.weight        transformer.h.0.mlp.c_proj.weight   ⚠️ 需要转置
    #   lm_head.weight                           (共享 wte.weight)                ⏭️ 跳过

    my_sd = model.state_dict()

    # 需要转置的 key 特征：来自 Conv1D 的权重
    # Conv1D 存储为 [in_features, out_features]，Linear 存储为 [out_features, in_features]
    transposed_keys = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    # 需要跳过的 key
    skip_keys = [
        "attn.masked_bias",    # HuggingFace 特有，你的模型用 register_buffer 不同
        "attn.bias",           # 同上
        "lm_head.weight",      # 你的模型共享 wte.weight
    ]

    print(f"\n🔄 开始映射权重...")

    loaded = 0
    skipped = 0

    for key in hf_sd:
        # 跳过不需要的 key
        if any(key.endswith(s) for s in skip_keys):
            skipped += 1
            continue

        # 检查 key 是否在你的模型中
        if key not in my_sd:
            print(f"  ⚠️ 跳过（你的模型中没有）: {key}")
            skipped += 1
            continue

        # 判断是否需要转置
        need_transpose = any(key.endswith(t) for t in transposed_keys)

        if need_transpose:
            # Conv1D → Linear: 转置权重
            assert hf_sd[key].shape[::-1] == my_sd[key].shape, \
                f"形状不匹配（转置后）: {key}: HF {hf_sd[key].shape} vs 你的 {my_sd[key].shape}"
            with torch.no_grad():
                my_sd[key].copy_(hf_sd[key].t())
        else:
            # 直接复制
            assert hf_sd[key].shape == my_sd[key].shape, \
                f"形状不匹配: {key}: HF {hf_sd[key].shape} vs 你的 {my_sd[key].shape}"
            with torch.no_grad():
                my_sd[key].copy_(hf_sd[key])

        loaded += 1

    print(f"\n  ✅ 成功加载: {loaded} 个参数张量")
    print(f"  ⏭️ 跳过: {skipped} 个")
    print(f"  📐 总参数量: {model.count_parameters():,}")

    model.to(device)
    model.eval()

    return model, cfg


def demo_generate(model, cfg, prompt="Hello, I am a language model,",
                  max_tokens=100, temperature=0.8, top_k=50):
    """用预训练模型生成文本"""
    from dataset import encode

    tokens = encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  参数: temperature={temperature}, top_k={top_k}")
    print(f"  {'─' * 50}")

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    result = decode(generated[0].tolist())
    print(f"  {result}")
    print(f"  {'─' * 50}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  🌟 加载 GPT-2 预训练权重")
    print("=" * 60)

    model, cfg = load_gpt2_from_huggingface("gpt2")

    print("\n" + "=" * 60)
    print("  📝 用预训练 GPT-2 生成文本（英文）")
    print("=" * 60)

    # 几个有趣的 prompt 来测试
    prompts = [
        "Hello, I am a language model,",
        "The meaning of life is",
        "In a distant galaxy,",
        "Once upon a time, there was a",
    ]

    for prompt in prompts:
        demo_generate(model, cfg, prompt=prompt, max_tokens=80)
        print()

    print("✅ 预训练权重加载和生成测试完成！")
    print("   如果生成的英文文本通顺且有意义，说明你的模型架构写对了！🎉")
