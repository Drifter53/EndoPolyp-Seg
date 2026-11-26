
from pathlib import Path

import torch

from model import UNet
from dataset import create_dataloaders


# ==================================
# 1. 设置设备
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================
# 2. 找到模型文件
# ==================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

checkpoint_path = (
    PROJECT_ROOT / "outputs" / "unet_epoch1.pth"
)


# ==================================
# 3. 加载模型检查点
# ==================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

print("保存的 Epoch：", checkpoint["epoch"])
print("训练 Loss：", checkpoint["train_loss"])
print("验证 Loss：", checkpoint["val_loss"])


# ==================================
# 4. 创建 U-Net
# ==================================

model = UNet().to(device)


# ==================================
# 5. 恢复训练好的参数
# ==================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# 切换到评估模式
model.eval()

print("模型参数加载成功！")


# ==================================
# 6. 读取一组验证数据
# ==================================

_, val_loader, _ = create_dataloaders(
    batch_size=4,
    num_workers=0
)

images, masks = next(iter(val_loader))

images = images.to(device)
masks = masks.to(device)


# ==================================
# 7. 使用加载的模型进行预测
# ==================================

with torch.no_grad():

    prediction = model(images)

    # Logits 转换为概率
    probability = torch.sigmoid(prediction)

    # 根据 0.5 阈值生成二值分割结果
    predicted_mask = (
        probability >= 0.5
    ).float()


# ==================================
# 8. 检查结果
# ==================================

print("输入图像形状：", images.shape)

print("真实 Mask 形状：", masks.shape)

print("预测 Logits 形状：", prediction.shape)

print("预测概率形状：", probability.shape)

print("最终预测 Mask 形状：", predicted_mask.shape)

print("模型加载与预测测试完成！")
