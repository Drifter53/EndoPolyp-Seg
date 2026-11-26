
from pathlib import Path

import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from transforms import apply_joint_augmentation


# ==================== 1. 项目路径配置 ====================

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 设置 Kvasir-SEG 数据集的根目录
DATA_DIR = PROJECT_ROOT / "data" / "Kvasir-SEG"

# 原始图像和分割掩码的存储路径
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

# 统一输入图像和掩码的尺寸（宽度，高度）
TARGET_SIZE = (256, 256)

# 数据划分文件所在目录
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# ==================== 2. ImageNet 归一化 ====================

# ImageNet 的 RGB 三通道均值
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).reshape(3, 1, 1)

# ImageNet 的 RGB 三通道标准差
IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).reshape(3, 1, 1)


# ImageNet 归一化函数
def normalize_image(image):

    mean = IMAGENET_MEAN.to(image.device)
    std = IMAGENET_STD.to(image.device)

    return (image - mean) / std


# ImageNet 反归一化函数
# 用于恢复图像，方便可视化
def denormalize_image(image):

    mean = IMAGENET_MEAN.to(image.device)
    std = IMAGENET_STD.to(image.device)

    return image * std + mean


# ==================== 3. 读取数据划分文件 ====================

def read_split(name):

    # 根据数据集名称找到对应的划分文件
    split_file = SPLIT_DIR / f"{name}.txt"

    # 保存图像及对应掩码的文件路径
    samples = []

    with open(split_file, "r", encoding="utf-8") as f:

        for line in f:

            # 去除首尾空白字符
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 获取图像文件名和掩码文件名
            image_name, mask_name = line.split("\t")

            # 拼接完整路径
            image_path = IMAGE_DIR / image_name
            mask_path = MASK_DIR / mask_name

            # 保存样本路径
            samples.append(
                (image_path, mask_path)
            )

    return samples


# ==================== 4. 定义 Dataset 类 ====================

class KvasirDataset(Dataset):

    def __init__(
        self,
        samples,
        augment=False,
        normalize_imagenet=False
    ):

        # 保存所有样本路径
        self.samples = samples

        # 是否启用数据增强
        self.augment = augment

        # 是否启用 ImageNet 归一化
        self.normalize_imagenet = normalize_imagenet

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        # 1. 获取当前样本路径
        image_path, mask_path = self.samples[index]

        # 2. 读取原始图像
        image = cv2.imread(str(image_path))

        # 读取单通道灰度 Mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        # 3. 检查读取是否成功
        if image is None or mask is None:

            raise ValueError(
                f"Image 或 Mask 读取失败："
                f"{image_path}, {mask_path}"
            )

        # 4. BGR 转 RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # 5. 检查原始图像和 Mask 的尺寸
        if image.shape[:2] != mask.shape[:2]:

            raise ValueError(
                f"Image 和 Mask 尺寸不一致："
                f"{image_path.name}"
            )

        # 6. 将图像调整为 256×256
        # 使用线性插值
        image = cv2.resize(
            image,
            TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # 将 Mask 调整为 256×256
        # 使用最近邻插值，避免产生新的标签值
        mask = cv2.resize(
            mask,
            TARGET_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # ==================== 数据增强 ====================

        # 仅训练集启用数据增强
        if self.augment:

            image, mask = apply_joint_augmentation(
                image,
                mask
            )

        # ==================== Tensor 转换 ====================

        # NumPy 转 PyTorch Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Image：HWC -> CHW
        image = image.permute(2, 0, 1)

        # 转换为 float32
        image = image.float()

        # Mask 增加通道维度
        mask = mask.unsqueeze(0).float()

        # ==================== 图像归一化 ====================

        # 第一步：将图像缩放至 [0, 1]
        image = image / 255.0

        # 第二步：根据配置执行 ImageNet 归一化
        if self.normalize_imagenet:

            image = normalize_image(image)

        # ==================== Mask 二值化 ====================

        # Mask 不进行 ImageNet 归一化
        # 前景为1，背景为0
        mask = (mask > 127).float()

        return image, mask


# ==================== 5. 创建三个 DataLoader ====================

def create_dataloaders(
    batch_size=4,
    num_workers=0,
    normalize_imagenet=False
):

    # 读取三个数据集的路径
    train_pairs = read_split("train")
    val_pairs = read_split("val")
    test_pairs = read_split("test")

    # 创建训练集
    train_dataset = KvasirDataset(
        train_pairs,
        augment=True,
        normalize_imagenet=normalize_imagenet
    )

    # 创建验证集
    val_dataset = KvasirDataset(
        val_pairs,
        augment=False,
        normalize_imagenet=normalize_imagenet
    )

    # 创建测试集
    test_dataset = KvasirDataset(
        test_pairs,
        augment=False,
        normalize_imagenet=normalize_imagenet
    )

    # 创建训练 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    # 创建验证 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    # 创建测试 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


# ==================== 6. 测试 Dataset 和 DataLoader ====================

if __name__ == "__main__":

    # 1. 创建三个 DataLoader
    # 本次测试开启 ImageNet 归一化
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
        normalize_imagenet=True
    )

    # 2. 检查数据增强状态
    print("\n========== 数据增强检查 ==========")

    print(
        "训练集是否启用增强：",
        train_loader.dataset.augment
    )

    print(
        "验证集是否启用增强：",
        val_loader.dataset.augment
    )

    print(
        "测试集是否启用增强：",
        test_loader.dataset.augment
    )

    # 3. 检查 ImageNet 归一化状态
    print("\n========== 归一化状态检查 ==========")

    print(
        "训练集是否启用 ImageNet 归一化：",
        train_loader.dataset.normalize_imagenet
    )

    print(
        "验证集是否启用 ImageNet 归一化：",
        val_loader.dataset.normalize_imagenet
    )

    print(
        "测试集是否启用 ImageNet 归一化：",
        test_loader.dataset.normalize_imagenet
    )

    # 4. 检查数据集数量
    print("\n========== 数据集数量检查 ==========")

    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(val_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))

    # 5. 检查 Batch 数量
    print("\n========== Batch 数量检查 ==========")

    print("训练集 Batch 数量：", len(train_loader))
    print("验证集 Batch 数量：", len(val_loader))
    print("测试集 Batch 数量：", len(test_loader))

    # 6. 读取一个 Batch
    images, masks = next(iter(train_loader))

    print("\n========== Tensor 形状检查 ==========")

    print("Images形状：", images.shape)
    print("Masks形状：", masks.shape)

    # 7. 检查归一化后的数值范围
    print("\n========== 数值范围检查 ==========")

    print("Images 最小值：", images.min().item())
    print("Images 最大值：", images.max().item())

    print("Masks 最小值：", masks.min().item())
    print("Masks 最大值：", masks.max().item())

    print("Masks 唯一值：", torch.unique(masks))

    # 8. 创建输出文件夹
    output_dir = PROJECT_ROOT / "outputs_resnet34"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. 保存当前 Batch 的可视化图片
    print("\n========== 可视化检查 ==========")

    for i in range(images.shape[0]):

        # 取出一张图像
        image_show = images[i].cpu()

        # 如果使用了 ImageNet 归一化，
        # 保存图片之前需要执行反归一化
        if train_loader.dataset.normalize_imagenet:

            image_show = denormalize_image(
                image_show
            )

        # 将像素值限制在 [0, 1]
        image_show = image_show.clamp(0, 1)

        # Image Tensor 转 NumPy
        # CHW -> HWC
        # [0, 1] -> [0, 255]
        image_np = (
            image_show.permute(1, 2, 0) * 255
        ).round().byte().numpy()

        # Mask Tensor 转 NumPy
        mask_np = (
            masks[i, 0].cpu() * 255
        ).byte().numpy()

        # RGB 转 BGR
        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        # 制作红色 Mask 叠加图
        overlay = image_bgr.copy()

        overlay[mask_np == 255] = (0, 0, 255)

        result = cv2.addWeighted(
            image_bgr, 0.7,
            overlay, 0.3,
            0
        )

        # Mask 转三通道
        mask_bgr = cv2.cvtColor(
            mask_np,
            cv2.COLOR_GRAY2BGR
        )

        # 拼接 Image、Mask、Overlay
        preview = cv2.hconcat([
            image_bgr,
            mask_bgr,
            result
        ])

        # 设置图片保存路径
        preview_path = (
            output_dir /
            f"augmentation_check_{i + 1}.jpg"
        )

        # 保存图片
        success = cv2.imwrite(
            str(preview_path),
            preview
        )

        if success:

            print(
                f"第{i + 1}张图片保存成功：",
                preview_path
            )

        else:

            print(
                f"第{i + 1}张图片保存失败！"
            )

    print("\nDataset 和 DataLoader 测试结束！")
