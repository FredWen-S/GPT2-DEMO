import torch

# 选设备：优先 CUDA，其次 Mac 的 MPS，最后 CPU
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
else 'cpu')
print('使用设备:', device)

x = torch.randint(0, 100, (2, 4))  # 形状 [2, 4]: 2 条序列，每条 4 个 token
print(x, x.shape)
