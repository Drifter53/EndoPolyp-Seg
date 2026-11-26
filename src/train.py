#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EndoPolyp-Seg: Original U-Net training script.

Restored as a separate script so it does not modify the ResNet34-U-Net or
Attention U-Net training code.

Original baseline configuration recovered from the earlier project script:
- model: src/model.py -> UNet()
- input: 256 x 256 RGB
- image range: [0, 1], no ImageNet normalization
- batch size: 4
- loss: CombinedLoss (BCEWithLogits + Dice)
- optimizer: Adam, lr=1e-4
- epochs: 100
- output: outputs/
"""

import argparse
import csv
import os
from pathlib import Path

import torch

from model import UNet
from losses import CombinedLoss
from dataset import create_dataloaders


INPUT_SIZE = (256, 256)
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
THRESHOLD = 0.5
SMOOTH = 1e-7


def calculate_batch_metrics(logits, masks, threshold=THRESHOLD, smooth=SMOOTH):
    """Return mean per-image Dice and IoU for the current batch."""
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
    return float(dice.mean().item()), float(iou.mean().item())


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=False)
        masks = masks.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        if logits.shape != masks.shape:
            raise RuntimeError(
                f"模型输出与 Mask 形状不一致：{tuple(logits.shape)} vs "
                f"{tuple(masks.shape)}"
            )

        loss = criterion(logits, masks)
        if not torch.isfinite(loss).item():
            raise RuntimeError("训练 Loss 出现 NaN 或 Inf。")

        loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        if batch_idx == 0:
            print("训练 Images：", tuple(images.shape))
            print("训练 Masks：", tuple(masks.shape))
            print("训练 Prediction：", tuple(logits.shape))

        print(
            f"训练 Batch [{batch_idx + 1}/{len(loader)}] "
            f"Loss: {loss.item():.4f}"
        )

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    if total_samples == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何样本。")

    return total_loss / total_samples


@torch.inference_mode()
def validate_one_epoch(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=False)
        masks = masks.to(device, non_blocking=False)

        logits = model(images)
        if logits.shape != masks.shape:
            raise RuntimeError(
                f"模型输出与 Mask 形状不一致：{tuple(logits.shape)} vs "
                f"{tuple(masks.shape)}"
            )

        loss = criterion(logits, masks)
        if not torch.isfinite(loss).item():
            raise RuntimeError("验证 Loss 出现 NaN 或 Inf。")

        dice, iou = calculate_batch_metrics(logits, masks)
        batch_size = int(images.shape[0])

        total_loss += float(loss.item()) * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_samples += batch_size

        if batch_idx == 0:
            print("验证 Images：", tuple(images.shape))
            print("验证 Prediction：", tuple(logits.shape))

        print(
            f"验证 Batch [{batch_idx + 1}/{len(loader)}] "
            f"Loss: {loss.item():.4f} "
            f"Dice: {dice:.4f} IoU: {iou:.4f}"
        )

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    if total_samples == 0:
        raise RuntimeError("验证 DataLoader 没有产生任何样本。")

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


def safe_torch_load(path):
    """Load a local checkpoint compatibly across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 格式不正确。")

    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    # Support a plain state_dict checkpoint.
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint 中没有找到 model_state_dict/state_dict。")


def validate_checkpoint_metadata(checkpoint, path):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint 格式错误：{path}")

    model_name = checkpoint.get("model_name")
    allowed_names = {None, "UNet", "OriginalUNet", "Original U-Net"}
    if model_name not in allowed_names:
        raise ValueError(
            f"{path} 记录的模型类型为 {model_name!r}，"
            "不是 Original U-Net。为避免误加载其他模型，已停止。"
        )

    input_size = checkpoint.get("input_size")
    if input_size is not None and tuple(input_size) != INPUT_SIZE:
        raise ValueError(
            f"Checkpoint 输入尺寸为 {input_size}，预期为 {INPUT_SIZE}。"
        )

    normalize_flag = checkpoint.get("normalize_imagenet")
    if normalize_flag is True:
        raise ValueError(
            "Checkpoint 标记为使用 ImageNet 归一化，"
            "与 Original U-Net 的 [0,1] 输入约定不一致。"
        )


def atomic_torch_save(payload, path):
    """Write checkpoint to a temporary file before replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def resolve_output_dir(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the standalone Original U-Net baseline."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="相对项目根目录或绝对路径；默认 outputs。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只训练/验证各 2 个 Batch，不保存权重或日志。",
    )
    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        help="仅当 latest_checkpoint.pth 不存在时，显式允许从 best_unet.pth 续训。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs、batch-size 和 lr 必须为正数。")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(project_root, args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练设备：", device)
    if device.type == "cuda":
        print("显卡名称：", torch.cuda.get_device_name(0))

    print("\n========== Original U-Net 配置 ==========")
    print("模型：model.UNet")
    print("输入尺寸：", INPUT_SIZE)
    print("Batch Size：", args.batch_size)
    print("ImageNet 归一化：False")
    print("Loss：CombinedLoss (BCE + Dice)")
    print("Optimizer：Adam")
    print("Learning Rate：", args.lr)
    print("计划 Epoch：", args.epochs)
    print("输出目录：", output_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_imagenet=False,
    )

    print("\n========== 数据集检查 ==========")
    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(val_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))
    print("训练 Batch：", len(train_loader))
    print("验证 Batch：", len(val_loader))
    print("测试 Batch：", len(test_loader))
    print("训练集数据增强：", getattr(train_loader.dataset, "augment", "unknown"))
    print("验证集数据增强：", getattr(val_loader.dataset, "augment", "unknown"))
    print("测试集数据增强：", getattr(test_loader.dataset, "augment", "unknown"))

    first_images, first_masks = next(iter(train_loader))
    print("首批 Images：", tuple(first_images.shape))
    print("首批 Masks：", tuple(first_masks.shape))
    if tuple(first_images.shape[-2:]) != INPUT_SIZE:
        raise RuntimeError(
            f"数据集输出尺寸为 {tuple(first_images.shape[-2:])}，"
            f"预期为 {INPUT_SIZE}。"
        )

    model = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量：{total_params:,} ({total_params / 1e6:.2f} M)")

    if args.smoke_test:
        print("\n========== Smoke Test ==========")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=2
        )
        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device, max_batches=2
        )
        print("\n快速测试通过（没有保存权重或日志）。")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Val Dice:   {val_dice:.6f}")
        print(f"Val IoU:    {val_iou:.6f}")
        return

    latest_path = output_dir / "latest_checkpoint.pth"
    initial_path = output_dir / "unet_epoch1.pth"
    best_path = output_dir / "best_unet.pth"
    history_path = output_dir / "training_history.csv"

    resume_path = None
    if latest_path.exists():
        resume_path = latest_path
    elif initial_path.exists():
        resume_path = initial_path
    elif args.resume_from_best and best_path.exists():
        resume_path = best_path

    # Protect the existing benchmark checkpoint from accidental fresh training.
    if resume_path is None and (
        best_path.exists() or history_path.exists()
    ):
        raise RuntimeError(
            f"发现已有 Original U-Net 实验文件，但没有可自动续训的 "
            f"latest_checkpoint.pth：\n{output_dir}\n"
            "为避免覆盖 outputs/best_unet.pth，脚本已停止。\n"
            "如果确实要从最佳权重继续训练，请添加 --resume-from-best；"
            "如果要从头开始新实验，请指定新的 --output-dir，"
            "例如 outputs_original_unet_newrun。"
        )

    start_epoch = 0
    best_val_loss = float("inf")
    last_metrics = None

    if resume_path is not None:
        checkpoint = safe_torch_load(resume_path)
        validate_checkpoint_metadata(checkpoint, resume_path)
        state_dict = extract_checkpoint_state(checkpoint)
        model.load_state_dict(state_dict, strict=True)

        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"{resume_path} 没有 optimizer_state_dict，无法可靠断点续训。"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("val_loss", float("inf")))
        last_metrics = checkpoint
        print("\n========== 断点续训 ==========")
        print("恢复检查点：", resume_path)
        print("已完成 Epoch：", start_epoch)

        if best_path.exists():
            best_checkpoint = safe_torch_load(best_path)
            validate_checkpoint_metadata(best_checkpoint, best_path)
            best_val_loss = min(
                best_val_loss,
                float(best_checkpoint.get("val_loss", float("inf"))),
            )
    else:
        print("\n没有历史检查点，从头训练 Original U-Net。")

    fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        with open(history_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if last_metrics is not None and all(k in last_metrics for k in fieldnames):
                writer.writerow({k: last_metrics[k] for k in fieldnames})

    if start_epoch >= args.epochs:
        print(
            f"\n检查点已完成 {start_epoch} Epoch，"
            f"不超过当前目标 {args.epochs}，无需继续训练。"
        )
        print("最佳模型：", best_path)
        return

    print("\n训练日志：", history_path)
    print("历史最佳 Validation Loss：", best_val_loss)

    for epoch_index in range(start_epoch, args.epochs):
        current_epoch = epoch_index + 1
        print("\n" + "=" * 58)
        print(f"Original U-Net Epoch [{current_epoch}/{args.epochs}]")
        print("=" * 58)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print("\n========== Epoch 结果 ==========")
        print(f"Epoch：{current_epoch}")
        print(f"Training Loss：{train_loss:.6f}")
        print(f"Validation Loss：{val_loss:.6f}")
        print(f"Validation Dice：{val_dice:.6f}")
        print(f"Validation IoU：{val_iou:.6f}")

        checkpoint_payload = {
            "model_name": "UNet",
            "input_size": INPUT_SIZE,
            "normalize_imagenet": False,
            "batch_size": args.batch_size,
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        }

        atomic_torch_save(checkpoint_payload, latest_path)
        print("最新检查点已保存：", latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            atomic_torch_save(checkpoint_payload, best_path)
            print("发现更好的模型，最佳权重已更新：", best_path)
        else:
            print("Validation Loss 未改善，保留历史最佳权重。")

        with open(history_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [current_epoch, train_loss, val_loss, val_dice, val_iou]
            )

    print("\n" + "=" * 58)
    print("Original U-Net 训练完成")
    print("=" * 58)
    print("最新检查点：", latest_path)
    print("最佳模型：", best_path)
    print("训练日志：", history_path)


if __name__ == "__main__":
    main()
