
import csv
import cv2
import torch

from pathlib import Path

from model_attention import UNet as AttentionUNet
from losses import CombinedLoss

from dataset import (
    create_dataloaders,
    denormalize_image
)


# ==========================================
# 1. 计算每张图片的 Dice 和 IoU
# ==========================================

def calculate_per_image_metrics(
    prediction,
    masks,
    threshold=0.5,
    smooth=1e-7
):

    prediction = (
        torch.sigmoid(prediction) >= threshold
    ).float()

    masks = (masks >= 0.5).float()

    prediction = prediction.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (
        prediction * masks
    ).sum(dim=1)

    pred_sum = prediction.sum(dim=1)
    mask_sum = masks.sum(dim=1)

    dice = (
        2 * intersection + smooth
    ) / (
        pred_sum + mask_sum + smooth
    )

    iou = (
        intersection + smooth
    ) / (
        pred_sum + mask_sum
        - intersection + smooth
    )

    return dice, iou


# ==========================================
# 2. 获取样本文件名
# ==========================================

def get_sample_names(dataset, index):

    image_path, mask_path = dataset.samples[index]

    return image_path.name, mask_path.name


# ==========================================
# 3. 制作分割结果对比图
# ==========================================

def create_comparison(
    image,
    mask,
    prediction,
    normalize_imagenet=True
):

    image = image.detach().cpu()

    # ImageNet 反归一化
    if normalize_imagenet:
        image = denormalize_image(image)

    image = image.clamp(0, 1)

    # CHW -> HWC
    image_np = (
        image.permute(1, 2, 0) * 255
    ).round().byte().numpy()

    # RGB -> BGR
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # Ground Truth
    mask_np = (
        (mask.detach().cpu()[0] >= 0.5) * 255
    ).byte().numpy()

    # Prediction
    pred_np = (
        (
            torch.sigmoid(
                prediction.detach().cpu()[0]
            ) >= 0.5
        ) * 255
    ).byte().numpy()

    # Mask 转三通道
    mask_bgr = cv2.cvtColor(
        mask_np,
        cv2.COLOR_GRAY2BGR
    )

    pred_bgr = cv2.cvtColor(
        pred_np,
        cv2.COLOR_GRAY2BGR
    )

    # 红色预测区域叠加图
    overlay = image_bgr.copy()

    overlay[pred_np == 255] = (0, 0, 255)

    result = cv2.addWeighted(
        image_bgr, 0.7,
        overlay, 0.3,
        0
    )

    panels = [
        image_bgr,
        mask_bgr,
        pred_bgr,
        result
    ]

    titles = [
        "Image",
        "Ground Truth",
        "Prediction",
        "Overlay"
    ]

    titled_panels = []

    for panel, title in zip(panels, titles):

        panel = cv2.copyMakeBorder(
            panel,
            30, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        cv2.putText(
            panel,
            title,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        titled_panels.append(panel)

    comparison = cv2.hconcat(titled_panels)

    return comparison


# ==========================================
# 4. 保存图片
# ==========================================

def save_image(path, image):

    success = cv2.imwrite(
        str(path),
        image
    )

    if not success:
        raise RuntimeError(f"图片保存失败：{path}")


# ==========================================
# 5. 保存每张图片的指标
# ==========================================

def save_metrics_csv(records, output_dir):

    all_path = output_dir / "all_image_metrics.csv"

    fieldnames = [
        "index",
        "image_name",
        "mask_name",
        "dice",
        "iou"
    ]

    with open(
        all_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            row = record.copy()

            row["dice"] = f"{record['dice']:.6f}"
            row["iou"] = f"{record['iou']:.6f}"

            writer.writerow(row)

    # Dice 从低到高排序
    sorted_records = sorted(
        records,
        key=lambda item: item["dice"]
    )

    worst_records = sorted_records[:20]

    worst_path = output_dir / "worst_20_images.csv"

    with open(
        worst_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "image_name",
            "mask_name",
            "dice",
            "iou"
        ])

        for rank, record in enumerate(
            worst_records,
            start=1
        ):

            writer.writerow([
                rank,
                record["image_name"],
                record["mask_name"],
                f"{record['dice']:.6f}",
                f"{record['iou']:.6f}"
            ])

    print("\n全部图片指标：", all_path)
    print("最差20张图片指标：", worst_path)

    return worst_records


# ==========================================
# 6. 保存最差20张图片的可视化
# ==========================================

def save_worst_visualizations(
    model,
    test_loader,
    device,
    worst_records,
    output_dir
):

    visual_dir = (
        output_dir / "worst_20_visualizations"
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    targets = {
        record["index"]: (rank, record)
        for rank, record in enumerate(
            worst_records,
            start=1
        )
    }

    sample_index = 0
    saved_count = 0

    model.eval()

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            for i in range(images.size(0)):

                if sample_index in targets:

                    rank, record = targets[sample_index]

                    comparison = create_comparison(
                        images[i],
                        masks[i],
                        predictions[i],
                        normalize_imagenet=(
                            test_loader.dataset.normalize_imagenet
                        )
                    )

                    comparison = cv2.copyMakeBorder(
                        comparison,
                        0, 30, 0, 0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                    information = (
                        f"Rank: {rank:02d} "
                        f"Dice: {record['dice']:.4f} "
                        f"IoU: {record['iou']:.4f} "
                        f"{record['image_name']}"
                    )

                    cv2.putText(
                        comparison,
                        information,
                        (8, comparison.shape[0] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    filename = f"rank_{rank:02d}.jpg"

                    save_image(
                        visual_dir / filename,
                        comparison
                    )

                    saved_count += 1

                    print(
                        f"最差第{rank:02d}张："
                        f"{record['image_name']} "
                        f"Dice={record['dice']:.4f}"
                    )

                sample_index += 1

            if saved_count == len(worst_records):
                break

    print("\n最差图片可视化保存完成：", saved_count)
    print("保存目录：", visual_dir)


# ==========================================
# 7. 完整测试流程
# ==========================================

def evaluate_test(
    model,
    test_loader,
    criterion,
    device,
    output_dir,
    num_visualizations=10
):

    model.eval()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    total_samples = 0
    sample_index = 0
    saved_count = 0

    all_records = []

    print("\n========== 开始 Attention U-Net 测试 ==========")

    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(
            test_loader
        ):

            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            prediction = model(images)

            assert prediction.shape == masks.shape

            # Loss
            loss = criterion(
                prediction,
                masks
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("测试 Loss 出现 NaN 或 Inf")

            # 每张图片的 Dice 和 IoU
            dice, iou = calculate_per_image_metrics(
                prediction,
                masks
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_dice += dice.sum().item()
            total_iou += iou.sum().item()

            total_samples += batch_size

            for i in range(batch_size):

                image_name, mask_name = get_sample_names(
                    test_loader.dataset,
                    sample_index
                )

                record = {
                    "index": sample_index,
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "dice": dice[i].item(),
                    "iou": iou[i].item()
                }

                all_records.append(record)

                # 保存前10张测试图片
                if saved_count < num_visualizations:

                    comparison = create_comparison(
                        images[i],
                        masks[i],
                        prediction[i],
                        normalize_imagenet=(
                            test_loader.dataset.normalize_imagenet
                        )
                    )

                    save_path = (
                        output_dir /
                        f"test_{saved_count + 1:03d}.jpg"
                    )

                    save_image(
                        save_path,
                        comparison
                    )

                    saved_count += 1

                sample_index += 1

            print(
                f"测试 Batch [{batch_idx + 1}/{len(test_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice.mean().item():.4f} "
                f"IoU: {iou.mean().item():.4f}"
            )

    assert total_samples == len(test_loader.dataset)

    # 平均指标
    average_loss = total_loss / total_samples
    average_dice = total_dice / total_samples
    average_iou = total_iou / total_samples

    # 保存测试指标
    metrics_path = output_dir / "test_metrics.txt"

    result_text = (
        f"Test Loss: {average_loss:.6f}\n"
        f"Test Dice: {average_dice:.6f}\n"
        f"Test IoU: {average_iou:.6f}\n"
        f"Test Samples: {total_samples}\n"
        f"Visualizations: {saved_count}\n"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result_text)

    # 保存每张图片的指标
    worst_records = save_metrics_csv(
        all_records,
        output_dir
    )

    # 保存最差20张图片
    save_worst_visualizations(
        model,
        test_loader,
        device,
        worst_records,
        output_dir
    )

    print("\n" + "=" * 50)
    print("Attention U-Net 测试完成！")
    print("=" * 50)

    print(result_text)
    print("测试指标保存位置：", metrics_path)

    return average_loss, average_dice, average_iou


# ==========================================
# 8. 主程序
# ==========================================

if __name__ == "__main__":

    # 8.1 设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("测试设备：", device)

    # 8.2 路径
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    output_dir = PROJECT_ROOT / "outputs_attention"

    result_dir = output_dir / "test_results"

    best_path = (
        output_dir / "best_attention_unet.pth"
    )

    # 检查最佳模型
    if not best_path.exists():
        raise FileNotFoundError(
            "没有找到 Attention U-Net 最佳模型："
            f"\n{best_path}"
        )

    # 8.3 创建测试 DataLoader
    _, _, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
        normalize_imagenet=True
    )

    print("测试集数量：", len(test_loader.dataset))
    print("测试集 Batch 数量：", len(test_loader))
    print(
        "ImageNet 归一化：",
        test_loader.dataset.normalize_imagenet
    )

    # 8.4 创建 Attention U-Net
    model = AttentionUNet().to(device)

    # 8.5 加载最佳模型
    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "检查点中没有 model_state_dict，"
            "请检查训练保存格式。"
        )

    if tuple(checkpoint.get("input_size", ())) != (256, 256):
        raise ValueError("检查点输入尺寸不是 256×256")

    if checkpoint.get("normalize_imagenet") is not True:
        raise ValueError("检查点的 ImageNet 归一化配置不正确")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\n========== Attention U-Net 最佳模型加载成功 ==========")
    print("模型路径：", best_path)
    print("模型类型：", checkpoint.get("model_name", "未记录"))
    print("最佳模型 Epoch：", checkpoint["epoch"])
    print("最佳 Validation Loss：", checkpoint["val_loss"])
    print("最佳 Validation Dice：", checkpoint["val_dice"])
    print("最佳 Validation IoU：", checkpoint["val_iou"])

    del checkpoint

    # 8.6 损失函数
    criterion = CombinedLoss()

    # 8.7 开始测试
    evaluate_test(
        model,
        test_loader,
        criterion,
        device,
        result_dir,
        num_visualizations=10
    )
