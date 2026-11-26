
import cv2
import torch


def apply_joint_augmentation(image, mask):
    """
    对 Image 和 Mask 执行同步数据增强。
    """

    # 1. 随机水平翻转
    if torch.rand(1).item() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # 2. 随机垂直翻转
    if torch.rand(1).item() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # 3. 随机旋转
    rotation = torch.randint(0, 4, (1,)).item()

    if rotation == 1:
        image = cv2.rotate(
            image, cv2.ROTATE_90_CLOCKWISE
        )
        mask = cv2.rotate(
            mask, cv2.ROTATE_90_CLOCKWISE
        )

    elif rotation == 2:
        image = cv2.rotate(
            image, cv2.ROTATE_180
        )
        mask = cv2.rotate(
            mask, cv2.ROTATE_180
        )

    elif rotation == 3:
        image = cv2.rotate(
            image, cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        mask = cv2.rotate(
            mask, cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image, mask
