
import torch
import torch.nn as nn


# ==================================
# BCE：二分类交叉熵损失
# ==================================
class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, masks):

        if prediction.shape != masks.shape:
            raise ValueError(
                "Prediction 和 Masks 形状不一致"
            )

        masks = masks.float()

        loss = self.loss_fn(prediction, masks)

        return loss


# ==================================
# DiceLoss：Dice 损失函数
# ==================================
class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, prediction, masks):

        if prediction.shape != masks.shape:
            raise ValueError(
                "Prediction 和 Masks 形状不一致"
            )

        masks = masks.float()

        # Logits 转换为概率
        prediction = torch.sigmoid(prediction)

        # 展平，保留 Batch 维度
        prediction = prediction.flatten(start_dim=1)
        masks = masks.flatten(start_dim=1)

        # 计算交集
        intersection = (prediction * masks).sum(dim=1)

        # 计算 Dice
        dice = (
            2 * intersection + self.smooth
        ) / (
            prediction.sum(dim=1)
            + masks.sum(dim=1)
            + self.smooth
        )

        # Dice Loss
        loss = 1 - dice.mean()

        return loss

# ==================================
# CombinedLoss：组合损失函数
# ==================================
class CombinedLoss(nn.Module):

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0
    ):
        super().__init__()

        # 创建两个损失函数
        self.bce = BCELoss()
        self.dice = DiceLoss()

        # 设置权重
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, prediction, masks):

        # 计算 BCE Loss
        bce_loss = self.bce(
            prediction,
            masks
        )

        # 计算 Dice Loss
        dice_loss = self.dice(
            prediction,
            masks
        )

        # 加权求和
        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss


# ==================================
# 测试两种损失函数
# ==================================
if __name__ == "__main__":

    # 模拟 U-Net 输出
    prediction = torch.randn(
        1, 1, 256, 256
    )

    # 模拟真实 Mask
    masks = torch.randint(
        0, 2,
        (1, 1, 256, 256)
    ).float()

    # 测试 BCE Loss
    bce_criterion = BCELoss()

    bce_loss = bce_criterion(
        prediction,
        masks
    )

    # 测试 Dice Loss
    dice_criterion = DiceLoss()

    dice_loss = dice_criterion(
        prediction,
        masks
    )

    print("Prediction形状：", prediction.shape)
    print("Masks形状：", masks.shape)

    print("BCE Loss：", bce_loss.item())
    print("Dice Loss：", dice_loss.item())

    # ==================================
    # 测试 Combined Loss
    # ==================================

    combined_criterion = CombinedLoss()

    combined_loss = combined_criterion(
        prediction,
        masks
    )

    print("Combined Loss：", combined_loss.item())
