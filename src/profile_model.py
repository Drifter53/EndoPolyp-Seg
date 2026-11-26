
import torch
import torch.nn as nn

from model import UNet


# ==================================
# 1. 创建 U-Net
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet().to(device)
model.eval()


# ==================================
# 2. 统计模型参数量
# ==================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


# ==================================
# 3. 准备 FLOPs 统计
# ==================================

layer_stats = []


def make_hook(layer_name):

    def hook(module, inputs, output):

        # 获取卷积核尺寸
        kh, kw = module.kernel_size

        if isinstance(module, nn.Conv2d):

            # 普通卷积的 MACs
            macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * kh * kw
            )

        elif isinstance(module, nn.ConvTranspose2d):

            # 转置卷积的 MACs（按常用公式估算）
            macs = (
                inputs[0].numel()
                * (module.out_channels // module.groups)
                * kh * kw
            )

        # 一个 MAC 按两次 FLOPs 计算
        flops = 2 * macs

        layer_stats.append({
            "name": layer_name,
            "type": module.__class__.__name__,
            "macs": macs,
            "flops": flops
        })

    return hook


# ==================================
# 4. 给所有卷积层注册 Hook
# ==================================

handles = []

for name, module in model.named_modules():

    if isinstance(
        module,
        (nn.Conv2d, nn.ConvTranspose2d)
    ):

        handle = module.register_forward_hook(
            make_hook(name)
        )

        handles.append(handle)


# ==================================
# 5. 执行一次前向传播
# ==================================

dummy_input = torch.randn(
    1, 3, 256, 256
).to(device)

with torch.no_grad():

    output = model(dummy_input)


# ==================================
# 6. 移除 Hook
# ==================================

for handle in handles:
    handle.remove()


# ==================================
# 7. 汇总计算量
# ==================================

total_macs = sum(
    item["macs"]
    for item in layer_stats
)

total_flops = sum(
    item["flops"]
    for item in layer_stats
)


# ==================================
# 8. 输出结果
# ==================================

print("\n========== U-Net 模型分析 ==========")

print("输入形状：", dummy_input.shape)
print("输出形状：", output.shape)

print("\n模型总参数量：", f"{total_params:,}")
print("可训练参数量：", f"{trainable_params:,}")

print(
    "参数量（百万）：",
    f"{total_params / 1_000_000:.3f} M"
)

print("\n卷积层数量：", len(layer_stats))

print("总 MACs：", f"{total_macs:,}")

print(
    "总 FLOPs：",
    f"{total_flops:,}"
)

print(
    "计算量（GFLOPs）：",
    f"{total_flops / 1_000_000_000:.3f}"
)


# ==================================
# 9. 输出每个卷积层的计算量
# ==================================

print("\n========== 各卷积层计算量 ==========")

for item in layer_stats:

    print(
        f"{item['name']:<35}"
        f"{item['type']:<18}"
        f"{item['flops'] / 1_000_000:.3f} MFLOPs"
    )

print("====================================")
