
import torch

from model import UNet


# ==================================
# 1. 创建 U-Net 模型
# ==================================

model = UNet()


# ==================================
# 2. 统计模型总参数量
# ==================================

total_params = sum(
    param.numel()
    for param in model.parameters()
)


# ==================================
# 3. 统计可训练参数量
# ==================================

trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)


# ==================================
# 4. 统计不可训练参数量
# ==================================

non_trainable_params = (
    total_params - trainable_params
)


# ==================================
# 5. 输出统计结果
# ==================================

print("========== U-Net 参数量统计 ==========")

print("模型总参数量：", f"{total_params:,}")

print("可训练参数量：", f"{trainable_params:,}")

print("不可训练参数量：", f"{non_trainable_params:,}")

print(
    "模型参数量（百万）：",
    f"{total_params / 1_000_000:.3f} M"
)

print("======================================")
