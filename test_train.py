"""快速训练测试 - 只跑 100 步验证训练流程"""
import config
# 覆盖配置为快速测试
config.max_iters = 100
config.eval_interval = 50
config.gen_max_tokens = 30

from train import main
main()
