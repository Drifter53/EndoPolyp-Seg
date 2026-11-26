
import csv
import cv2
import numpy as np
import torch

from pathlib import Path

from model import UNet as OriginalUNet
from model_resnet import ResNet34UNet
from model_attention import UNet as AttentionUNet

from dataset import create_dataloaders, normalize_image
from losses import CombinedLoss


# ==========================================
# 1. 基础配置
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 三个模型的权重路径
UNET_PATH = PROJECT_ROOT / "outputs" / "best_unet.pth"

RESNET_PATH = (
    PROJECT_ROOT / "outputs_resnet34" / "best_unet.pth"
)

ATTENTION_PATH = (
    PROJECT_ROOT / "outputs_attention"
    / "best_attention_unet.pth"
)

# 对照实验结果保存路径
OUTPUT_DIR = PROJECT_ROOT / "outputs_comparison"

# 测试参数
BATCH_SIZE = 4
NUM_WORKERS = 0

# 保存前多少张测试图像的可视化结果
NUM_VISUALIZATIONS = 20

# 预测二值化阈值
THRESHOLD = 0.5

# Dice 和 IoU 平滑参数
SMOOTH = 1e-7

# 三个模型的输入尺寸
INPUT_SIZE = (256, 256)


# ==========================================
# 2. 加载模型权重
# ==========================================

def safe_torch_load(path):

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

    return checkpoint


def load_model_weights(
    model,
    checkpoint_path,
    expected_model_name=None,
    expected_normalize=False
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"没有找到模型权重：{checkpoint_path}"
        )

    checkpoint = safe_torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):

        raise ValueError(
            f"检查点格式错误：{checkpoint_path}"
        )

    # 兼容完整检查点和纯 state_dict
    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    if not isinstance(state_dict, dict):

        raise ValueError("模型参数格式不正确！")

    # 检查模型名称
    if expected_model_name is not None:

        actual_name = checkpoint.get("model_name")

        if (
            actual_name is not None
            and actual_name != expected_model_name
        ):

            raise ValueError(
                f"模型类型不匹配：{actual_name}"
            )

    # 检查输入尺寸
    actual_size = checkpoint.get("input_size")

    if actual_size is not None:

        if tuple(actual_size) != INPUT_SIZE:

            raise ValueError(
                f"模型输入尺寸错误：{actual_size}"
            )

    else:

        print(
            f"[提示] {checkpoint_path.name} "
            "没有保存 input_size 元数据。"
        )

    # 检查归一化配置
    actual_normalize = checkpoint.get(
        "normalize_imagenet"
    )

    if actual_normalize is not None:

        if bool(actual_normalize) != expected_normalize:

            raise ValueError(
                f"归一化配置不匹配：{checkpoint_path}"
            )

    else:

        print(
            f"[提示] {checkpoint_path.name} "
            f"没有保存归一化元数据，"
            f"按照训练脚本使用 normalize_imagenet="
            f"{expected_normalize}"
        )

    # 严格加载模型参数
    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(f"\n模型加载成功：{checkpoint_path}")

    if "epoch" in checkpoint:

        print("训练 Epoch：", checkpoint["epoch"])

    if "val_loss" in checkpoint:

        print("最佳验证 Loss：", checkpoint["val_loss"])

    return model


# ==========================================
# 3. 计算每张图片的 Dice 和 IoU
# ==========================================

def calculate_per_image_metrics(
    prediction,
    masks
):

    # Logits 转概率并进行二值化
    prediction = (
        torch.sigmoid(prediction) >= THRESHOLD
    ).float()

    masks = (
        masks >= 0.5
    ).float()

    # 保留 Batch 维度，其余维度展开
    prediction_flat = prediction.flatten(
        start_dim=1
    )

    masks_flat = masks.flatten(
        start_dim=1
    )

    # 计算交集
    intersection = (
        prediction_flat * masks_flat
    ).sum(dim=1)

    pred_sum = prediction_flat.sum(dim=1)

    mask_sum = masks_flat.sum(dim=1)

    # Dice
    dice = (
        2 * intersection + SMOOTH
    ) / (
        pred_sum + mask_sum + SMOOTH
    )

    # IoU
    iou = (
        intersection + SMOOTH
    ) / (
        pred_sum + mask_sum
        - intersection + SMOOTH
    )

    return dice, iou, prediction


# ==========================================
# 4. 可视化辅助函数
# ==========================================

def add_title(image, title):

    height, width = image.shape[:2]

    header = np.zeros(
        (30, width, 3),
        dtype=np.uint8
    )

    cv2.putText(
        header,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return cv2.vconcat([
        header,
        image
    ])


def create_montage(
    image,
    mask,
    predictions
):

    # 原始图像 [C,H,W] -> BGR
    image_np = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image_np = (
        np.clip(image_np, 0, 1) * 255
    ).round().astype(np.uint8)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # 真实 Mask
    mask_np = (
        mask.detach().cpu()[0].numpy() >= 0.5
    ).astype(np.uint8) * 255

    mask_bgr = cv2.cvtColor(
        mask_np,
        cv2.COLOR_GRAY2BGR
    )

    panels = [
        add_title(image_bgr, "Original Image"),
        add_title(mask_bgr, "Ground Truth")
    ]

    # 三个模型的预测结果
    titles = {
        "unet": "Original U-Net",
        "resnet34_unet": "ResNet34 U-Net",
        "attention_unet": "Attention U-Net"
    }

    for name in [
        "unet",
        "resnet34_unet",
        "attention_unet"
    ]:

        pred_np = (
            predictions[name]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8) * 255
        )

        pred_bgr = cv2.cvtColor(
            pred_np,
            cv2.COLOR_GRAY2BGR
        )

        panels.append(
            add_title(
                pred_bgr,
                titles[name]
            )
        )

    # 五张图水平拼接
    return cv2.hconcat(panels)


# ==========================================
# 5. 保存图片
# ==========================================

def save_image(path, image):

    success = cv2.imwrite(
        str(path),
        image
    )

    if not success:

        raise RuntimeError(
            f"图片保存失败：{path}"
        )


# ==========================================
# 6. 保存 CSV
# ==========================================

def save_csv(path, records, fieldnames):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)


# ==========================================
# 7. 主测试程序
# ==========================================

def main():

    # --------------------------------------
    # 7.1 设置设备
    # --------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("三模型图像分割对照实验")
    print("=" * 55)

    print("测试设备：", device)

    if device.type == "cuda":

        print(
            "显卡名称：",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------
    # 7.2 创建测试集
    # --------------------------------------

    # 先读取未进行 ImageNet 归一化的图像。
    # 三个模型共用这一份测试集。

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize_imagenet=False
    )

    if len(test_loader.dataset) == 0:

        raise RuntimeError("测试集为空！")

    if test_loader.dataset.augment:

        raise RuntimeError(
            "测试集不应该启用数据增强！"
        )

    print("\n测试集数量：", len(test_loader.dataset))
    print("测试 Batch 数量：", len(test_loader))

    print("输入尺寸：", INPUT_SIZE)

    print("测试集 ImageNet 归一化：False")

    # --------------------------------------
    # 7.3 创建三个模型
    # --------------------------------------

    models = {

        "unet": OriginalUNet().to(device),

        "resnet34_unet": ResNet34UNet(
            pretrained=False
        ).to(device),

        "attention_unet": AttentionUNet().to(device)

    }

    # --------------------------------------
    # 7.4 加载三个模型权重
    # --------------------------------------

    models["unet"] = load_model_weights(
        models["unet"],
        UNET_PATH,
        expected_normalize=False
    )

    models["resnet34_unet"] = load_model_weights(
        models["resnet34_unet"],
        RESNET_PATH,
        expected_model_name="ResNet34UNet",
        expected_normalize=True
    )

    models["attention_unet"] = load_model_weights(
        models["attention_unet"],
        ATTENTION_PATH,
        expected_model_name="AttentionUNet",
        expected_normalize=True
    )

    for model in models.values():

        model.eval()

    print("\n三个模型均已加载完成！")

    # --------------------------------------
    # 7.5 创建损失函数
    # --------------------------------------

    criterion = CombinedLoss()

    # --------------------------------------
    # 7.6 创建输出文件夹
    # --------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    visual_dir = OUTPUT_DIR / "visualizations"

    visual_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # 7.7 初始化评价记录
    # --------------------------------------

    metric_store = {

        name: {
            "dice": [],
            "iou": [],
            "loss_sum": 0.0
        }

        for name in models

    }

    all_records = []

    sample_index = 0

    # --------------------------------------
    # 7.8 开始测试
    # --------------------------------------

    with torch.inference_mode():

        for batch_idx, (images, masks) in enumerate(
            test_loader
        ):

            # images 当前为 RGB、[0,1]
            images = images.to(device)

            masks = masks.to(device)

            # 为另外两个模型生成 ImageNet 归一化输入
            images_imagenet = normalize_image(images)

            # 三个模型分别前向传播
            outputs = {

                "unet": models["unet"](images),

                "resnet34_unet": models[
                    "resnet34_unet"
                ](images_imagenet),

                "attention_unet": models[
                    "attention_unet"
                ](images_imagenet)

            }

            batch_size = images.size(0)

            batch_metrics = {}

            batch_predictions = {}

            # ----------------------------------
            # 计算三个模型的指标
            # ----------------------------------

            for name, prediction in outputs.items():

                # 检查输出形状
                if prediction.shape != masks.shape:

                    raise ValueError(
                        f"{name} 输出尺寸不正确："
                        f"{prediction.shape}，"
                        f"真实标签尺寸：{masks.shape}"
                    )

                # 计算 Combined Loss
                loss = criterion(
                    prediction,
                    masks
                )

                if not torch.isfinite(loss).item():

                    raise RuntimeError(
                        f"{name} 测试 Loss 异常！"
                    )

                # 计算逐图 Dice、IoU
                dice, iou, pred_mask = (
                    calculate_per_image_metrics(
                        prediction,
                        masks
                    )
                )

                metric_store[name]["loss_sum"] += (
                    loss.item() * batch_size
                )

                metric_store[name]["dice"].extend(
                    dice.cpu().tolist()
                )

                metric_store[name]["iou"].extend(
                    iou.cpu().tolist()
                )

                batch_metrics[name] = {
                    "dice": dice,
                    "iou": iou
                }

                batch_predictions[name] = pred_mask.cpu()

            # ----------------------------------
            # 保存每张图片的评价指标
            # ----------------------------------

            for i in range(batch_size):

                image_path, mask_path = (
                    test_loader.dataset.samples[
                        sample_index
                    ]
                )

                record = {

                    "index": sample_index,

                    "image_name": image_path.name,

                    "mask_name": mask_path.name

                }

                for name in models:

                    record[f"{name}_dice"] = (
                        batch_metrics[name]["dice"][i].item()
                    )

                    record[f"{name}_iou"] = (
                        batch_metrics[name]["iou"][i].item()
                    )

                all_records.append(record)

                # ----------------------------------
                # 保存相同测试图像的五图对比
                # ----------------------------------

                if sample_index < NUM_VISUALIZATIONS:

                    predictions = {

                        name: batch_predictions[name][i, 0]

                        for name in models

                    }

                    montage = create_montage(
                        images[i],
                        masks[i],
                        predictions
                    )

                    save_path = (
                        visual_dir
                        / f"{sample_index + 1:04d}_"
                        f"{image_path.stem}.jpg"
                    )

                    save_image(
                        save_path,
                        montage
                    )

                sample_index += 1

            print(
                f"测试 Batch "
                f"[{batch_idx + 1}/{len(test_loader)}] "
                "完成"
            )

    # --------------------------------------
    # 7.9 检查测试样本数量
    # --------------------------------------

    total_samples = len(test_loader.dataset)

    if sample_index != total_samples:

        raise RuntimeError(
            "测试样本数量不一致！"
        )

    print("\n全部测试图像处理完成！")

    # --------------------------------------
    # 7.10 计算最终平均指标
    # --------------------------------------

    summary_records = []

    for name in models:

        dice_values = np.array(
            metric_store[name]["dice"]
        )

        iou_values = np.array(
            metric_store[name]["iou"]
        )

        average_loss = (
            metric_store[name]["loss_sum"]
            / total_samples
        )

        average_dice = float(
            dice_values.mean()
        )

        average_iou = float(
            iou_values.mean()
        )

        dice_std = float(
            dice_values.std()
        )

        iou_std = float(
            iou_values.std()
        )

        if name == "unet":

            model_name = "Original U-Net"
            checkpoint_path = UNET_PATH
            normalize_flag = False

        elif name == "resnet34_unet":

            model_name = "ResNet34-U-Net"
            checkpoint_path = RESNET_PATH
            normalize_flag = True

        else:

            model_name = "Attention U-Net"
            checkpoint_path = ATTENTION_PATH
            normalize_flag = True

        summary_records.append({

            "model": model_name,

            "checkpoint": str(
                checkpoint_path.relative_to(PROJECT_ROOT)
            ),

            "input_size": "256x256",

            "normalize_imagenet": normalize_flag,

            "test_samples": total_samples,

            "mean_loss": average_loss,

            "mean_dice": average_dice,

            "std_dice": dice_std,

            "mean_iou": average_iou,

            "std_iou": iou_std

        })

    # --------------------------------------
    # 7.11 保存逐图指标
    # --------------------------------------

    per_image_fields = [
        "index",
        "image_name",
        "mask_name"
    ]

    for name in models:

        per_image_fields.extend([
            f"{name}_dice",
            f"{name}_iou"
        ])

    save_csv(
        OUTPUT_DIR / "per_image_metrics.csv",
        all_records,
        per_image_fields
    )

    # --------------------------------------
    # 7.12 保存总体指标
    # --------------------------------------

    summary_fields = [
        "model",
        "checkpoint",
        "input_size",
        "normalize_imagenet",
        "test_samples",
        "mean_loss",
        "mean_dice",
        "std_dice",
        "mean_iou",
        "std_iou"
    ]

    save_csv(
        OUTPUT_DIR / "summary.csv",
        summary_records,
        summary_fields
    )

    # --------------------------------------
    # 7.13 保存文字报告
    # --------------------------------------

    report_path = OUTPUT_DIR / "summary.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("三模型图像分割对照实验\n")

        f.write("=" * 50 + "\n")

        f.write(f"测试样本数量：{total_samples}\n")

        f.write(f"输入尺寸：{INPUT_SIZE}\n")

        f.write(f"预测阈值：{THRESHOLD}\n\n")

        for result in summary_records:

            f.write(
                f"模型：{result['model']}\n"
            )

            f.write(
                f"Test Loss：{result['mean_loss']:.6f}\n"
            )

            f.write(
                f"Mean Dice：{result['mean_dice']:.6f}\n"
            )

            f.write(
                f"Dice Std：{result['std_dice']:.6f}\n"
            )

            f.write(
                f"Mean IoU：{result['mean_iou']:.6f}\n"
            )

            f.write(
                f"IoU Std：{result['std_iou']:.6f}\n"
            )

            f.write("-" * 50 + "\n")

    # --------------------------------------
    # 7.14 打印最终结果
    # --------------------------------------

    print("\n" + "=" * 60)

    print("三模型对照实验完成！")

    print("=" * 60)

    for result in summary_records:

        print("\n模型：", result["model"])

        print(
            f"Test Loss：{result['mean_loss']:.6f}"
        )

        print(
            f"Mean Dice：{result['mean_dice']:.6f}"
        )

        print(
            f"Mean IoU：{result['mean_iou']:.6f}"
        )

    print("\n结果保存目录：", OUTPUT_DIR)

    print("\n总体指标：", OUTPUT_DIR / "summary.csv")

    print(
        "逐图指标：",
        OUTPUT_DIR / "per_image_metrics.csv"
    )

    print("可视化结果：", visual_dir)

    print("\n实验结束！")


# ==========================================
# 8. 程序入口
# ==========================================

if __name__ == "__main__":

    main()
