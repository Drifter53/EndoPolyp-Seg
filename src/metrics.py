
import torch


# ==================================
# 1. Dice Score
# ==================================

def dice_score(
    prediction,
    masks,
    threshold=0.5,
    smooth=1.0
):

    if prediction.shape != masks.shape:
        raise ValueError("Prediction 和 Masks 形状不一致")

    # Logits 转换为概率
    prediction = torch.sigmoid(prediction)

    # 转换为二值 Mask
    prediction = (prediction > threshold).float()
    masks = (masks > 0.5).float()

    # 展平空间维度，保留 Batch
    prediction = prediction.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    # 计算交集
    intersection = (prediction * masks).sum(dim=1)

    # 计算 Dice
    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum(dim=1)
        + masks.sum(dim=1)
        + smooth
    )

    return dice.mean()


# ==================================
# 2. IoU Score
# ==================================

def iou_score(
    prediction,
    masks,
    threshold=0.5,
    smooth=1.0
):

    if prediction.shape != masks.shape:
        raise ValueError("Prediction 和 Masks 形状不一致")

    # Logits 转换为概率
    prediction = torch.sigmoid(prediction)

    # 转换为二值 Mask
    prediction = (prediction > threshold).float()
    masks = (masks > 0.5).float()

    # 展平空间维度
    prediction = prediction.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    # 计算交集
    intersection = (prediction * masks).sum(dim=1)

    # 计算并集
    union = (
        prediction.sum(dim=1)
        + masks.sum(dim=1)
        - intersection
    )

    # 计算 IoU
    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


# ==================================
# 3. 单独测试 metrics.py
# ==================================

if __name__ == "__main__":

    # 模拟模型输出 Logits
    prediction = torch.tensor([
        [[[
            10.0, 10.0, -10.0, -10.0
        ]]]
    ])

    # 模拟真实 Mask
    masks = torch.tensor([
        [[[
            1.0, 0.0, 1.0, 0.0
        ]]]
    ])

    # 计算 Dice
    dice = dice_score(
        prediction,
        masks
    )

    # 计算 IoU
    iou = iou_score(
        prediction,
        masks
    )

    print("Dice Score：", dice.item())
    print("IoU Score：", iou.item())
