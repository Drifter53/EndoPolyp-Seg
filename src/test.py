#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EndoPolyp-Seg: Original U-Net test/evaluation script.

Loads outputs/best_unet.pth and evaluates the unchanged test split.
The preprocessing is the Original U-Net convention: RGB float image in [0,1],
without ImageNet normalization.
"""

import argparse
import csv
from pathlib import Path

import cv2
import torch

from model import UNet
from losses import CombinedLoss
from dataset import create_dataloaders


THRESHOLD = 0.5
SMOOTH = 1e-7


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 格式不正确。")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint 中没有找到模型参数。")


def load_original_checkpoint(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"没有找到 Original U-Net 权重：{checkpoint_path}\n"
            "请确认 outputs/best_unet.pth 存在。"
        )

    checkpoint = safe_torch_load(checkpoint_path)

    if isinstance(checkpoint, dict):
        model_name = checkpoint.get("model_name")
        allowed_names = {None, "UNet", "OriginalUNet", "Original U-Net"}
        if model_name not in allowed_names:
            raise ValueError(
                f"权重记录的模型类型为 {model_name!r}，"
                "不是 Original U-Net。"
            )

        input_size = checkpoint.get("input_size")
        if input_size is not None and tuple(input_size) != (256, 256):
            raise ValueError(f"Checkpoint 输入尺寸不匹配：{input_size}")

        normalize_flag = checkpoint.get("normalize_imagenet")
        if normalize_flag is True:
            raise ValueError(
                "该权重标记为 ImageNet 归一化训练，"
                "不能按 Original U-Net 的预处理方式测试。"
            )

    state_dict = extract_state_dict(checkpoint)

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def calculate_per_image_metrics(logits, masks, threshold=THRESHOLD, smooth=SMOOTH):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = (masks >= 0.5).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (predictions * targets).sum(dim=1)
    pred_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        pred_sum + target_sum + smooth
    )
    iou = (intersection + smooth) / (
        pred_sum + target_sum - intersection + smooth
    )
    return dice, iou


def create_comparison(image, mask, logits):
    """Create Image / Ground Truth / Prediction / Overlay panels."""
    image = image.detach().cpu().clamp(0, 1)
    image_np = (
        image.permute(1, 2, 0) * 255.0
    ).round().byte().numpy()
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mask_np = (
        (mask.detach().cpu()[0] >= 0.5) * 255
    ).byte().numpy()

    pred_np = (
        (torch.sigmoid(logits.detach().cpu()[0]) >= THRESHOLD) * 255
    ).byte().numpy()

    mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)

    overlay = image_bgr.copy()
    overlay[pred_np == 255] = (0, 0, 255)
    overlay_bgr = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

    panels = [
        image_bgr,
        mask_bgr,
        pred_bgr,
        overlay_bgr,
    ]
    titles = ["Image", "Ground Truth", "Prediction", "Overlay"]
    titled = []

    for panel, title in zip(panels, titles):
        panel = cv2.copyMakeBorder(
            panel, 30, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        cv2.putText(
            panel,
            title,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        titled.append(panel)

    return cv2.hconcat(titled)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"图片保存失败：{path}")


def save_csv(path, records, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_worst_visualizations(model, test_loader, device, worst_records, visual_dir):
    if not worst_records:
        return

    targets = {
        int(record["index"]): (rank, record)
        for rank, record in enumerate(worst_records, start=1)
    }

    visual_dir.mkdir(parents=True, exist_ok=True)
    sample_index = 0
    saved_count = 0
    model.eval()

    with torch.inference_mode():
        for images, masks in test_loader:
            logits = model(images.to(device))

            for item_index in range(images.shape[0]):
                if sample_index in targets:
                    rank, record = targets[sample_index]
                    comparison = create_comparison(
                        images[item_index],
                        masks[item_index],
                        logits[item_index],
                    )

                    comparison = cv2.copyMakeBorder(
                        comparison,
                        0, 30, 0, 0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    info = (
                        f"Rank: {rank:02d}  "
                        f"Dice: {record['dice']:.4f}  "
                        f"IoU: {record['iou']:.4f}  "
                        f"{record['image_name']}"
                    )
                    cv2.putText(
                        comparison,
                        info,
                        (8, comparison.shape[0] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    save_image(visual_dir / f"rank_{rank:02d}.jpg", comparison)
                    saved_count += 1
                sample_index += 1

            if saved_count == len(worst_records):
                break

    print(f"最差样本可视化：{saved_count} 张 -> {visual_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Original U-Net checkpoint on the test split."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visualizations", type=int, default=10)
    parser.add_argument("--worst-k", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_unet.pth",
        help="相对项目根目录或绝对权重路径。",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="outputs/test_results_original_unet",
        help="相对项目根目录或绝对结果目录。",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="可选：只测试前 N 个 Batch 做快速检查；正式测试不要设置。",
    )
    return parser.parse_args()


def resolve_path(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正数，num-workers 不能为负数。")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches 必须为正数。")

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    result_dir = resolve_path(project_root, args.result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("测试设备：", device)
    if device.type == "cuda":
        print("显卡名称：", torch.cuda.get_device_name(0))

    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_imagenet=False,
    )

    if getattr(test_loader.dataset, "augment", False):
        raise RuntimeError("测试集不应启用数据增强。")
    if getattr(test_loader.dataset, "normalize_imagenet", False):
        raise RuntimeError("Original U-Net 测试集不应使用 ImageNet 归一化。")

    model = UNet().to(device)
    checkpoint = load_original_checkpoint(model, checkpoint_path)
    model.eval()

    print("\n========== Original U-Net 权重加载成功 ==========")
    print("权重路径：", checkpoint_path)
    if isinstance(checkpoint, dict):
        print("Checkpoint Epoch：", checkpoint.get("epoch", "未记录"))
        print("Validation Loss：", checkpoint.get("val_loss", "未记录"))
    print("输入尺寸：256 x 256")
    print("ImageNet 归一化：False")
    print("测试集图片数：", len(test_loader.dataset))
    print("测试集 Batch 数：", len(test_loader))

    criterion = CombinedLoss()
    all_records = []
    total_loss = 0.0
    total_samples = 0
    sample_index = 0
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== 开始测试 ==========")
    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images_gpu = images.to(device)
            masks_gpu = masks.to(device)

            logits = model(images_gpu)
            if logits.shape != masks_gpu.shape:
                raise RuntimeError(
                    f"输出与标签形状不一致：{tuple(logits.shape)} vs "
                    f"{tuple(masks_gpu.shape)}"
                )

            loss = criterion(logits, masks_gpu)
            if not torch.isfinite(loss).item():
                raise RuntimeError("测试 Loss 出现 NaN 或 Inf。")

            dice_values, iou_values = calculate_per_image_metrics(
                logits, masks_gpu
            )

            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            for i in range(batch_size):
                image_path, mask_path = test_loader.dataset.samples[sample_index]
                record = {
                    "index": sample_index,
                    "image_name": image_path.name,
                    "mask_name": mask_path.name,
                    "dice": float(dice_values[i].item()),
                    "iou": float(iou_values[i].item()),
                }
                all_records.append(record)

                if sample_index < args.num_visualizations:
                    comparison = create_comparison(
                        images[i],
                        masks[i],
                        logits[i],
                    )
                    save_image(
                        result_dir / f"test_{sample_index + 1:03d}.jpg",
                        comparison,
                    )

                sample_index += 1

            print(
                f"测试 Batch [{batch_idx + 1}/{len(test_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice_values.mean().item():.4f} "
                f"IoU: {iou_values.mean().item():.4f}"
            )

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break

    if args.max_batches is None and total_samples != len(test_loader.dataset):
        raise RuntimeError(
            f"测试样本数不一致：实际 {total_samples}，"
            f"数据集 {len(test_loader.dataset)}"
        )
    if total_samples == 0:
        raise RuntimeError("测试集没有产生任何样本。")

    average_loss = total_loss / total_samples
    average_dice = sum(r["dice"] for r in all_records) / len(all_records)
    average_iou = sum(r["iou"] for r in all_records) / len(all_records)

    all_csv = result_dir / "all_image_metrics.csv"
    save_csv(
        all_csv,
        all_records,
        ["index", "image_name", "mask_name", "dice", "iou"],
    )

    worst_records = sorted(all_records, key=lambda r: r["dice"])[:args.worst_k]
    worst_csv = result_dir / f"worst_{len(worst_records)}_images.csv"
    save_csv(
        worst_csv,
        [
            {
                "rank": rank,
                "image_name": record["image_name"],
                "mask_name": record["mask_name"],
                "dice": f"{record['dice']:.6f}",
                "iou": f"{record['iou']:.6f}",
            }
            for rank, record in enumerate(worst_records, start=1)
        ],
        ["rank", "image_name", "mask_name", "dice", "iou"],
    )

    worst_visual_dir = result_dir / f"worst_{len(worst_records)}_visualizations"
    save_worst_visualizations(
        model, test_loader, device, worst_records, worst_visual_dir
    )

    status = "FULL TEST SET" if args.max_batches is None else "PARTIAL TEST"
    report = (
        f"Original U-Net Test Report ({status})\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Device: {device}\n"
        f"Input Size: 256x256\n"
        f"Batch Size: {args.batch_size}\n"
        f"ImageNet Normalization: False\n"
        f"Test Samples: {total_samples}\n"
        f"Test Loss: {average_loss:.6f}\n"
        f"Test Dice: {average_dice:.6f}\n"
        f"Test IoU: {average_iou:.6f}\n"
        f"First Visualizations: {min(total_samples, args.num_visualizations)}\n"
    )
    metrics_path = result_dir / "test_metrics.txt"
    metrics_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 58)
    print("Original U-Net 测试完成")
    print("=" * 58)
    print(report)
    print("逐图指标：", all_csv)
    print("最差样本指标：", worst_csv)
    print("结果目录：", result_dir)


if __name__ == "__main__":
    main()
