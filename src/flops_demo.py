
# ==================================
# 计算 U-Net 第一层卷积的 FLOPs
# ==================================

# 输入图像参数
batch_size = 1
in_channels = 3

# 输出通道
out_channels = 64

# 图像尺寸
height = 256
width = 256

# 卷积核尺寸
kernel_size = 3

# 计算 MACs
macs = (
    batch_size
    * out_channels
    * height
    * width
    * in_channels
    * kernel_size
    * kernel_size
)

# 一个 MAC 按两次 FLOPs 计算
flops = macs * 2

print("========== 第一层卷积计算量 ==========")

print("MACs：", f"{macs:,}")
print("FLOPs：", f"{flops:,}")

print(
    "计算量（MFLOPs）：",
    f"{flops / 1_000_000:.2f}"
)

print("======================================")
