
import csv
import torch
from pathlib import Path

from model_resnet import ResNet34UNet
from losses import CombinedLoss
from dataset import create_dataloaders
from metrics import dice_score, iou_score


# ==========================================
# 1. 训练参数配置
# ==========================================

BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 101

# 编码器学习率
ENCODER_LR = 1e-5

# 解码器和输出层学习率
DECODER_LR = 1e-4

# 快速测试开关
# True：只测试少量 Batch，不保存模型
# False：正式进行100轮训练
SMOKE_TEST = False


# ==========================================
# 2. 训练一个 Epoch
# ==========================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    max_batches=None
):

    # 设置训练模式
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        # 将数据移动到对应设备
        images = images.to(device)
        masks = masks.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        prediction = model(images)

        # 检查预测与标签形状
        assert prediction.shape == masks.shape

        # 计算损失
        loss = criterion(prediction, masks)

        # 检查 Loss 是否正常
        if not torch.isfinite(loss).item():

            raise RuntimeError(
                "训练 Loss 出现 NaN 或 Inf"
            )

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 当前 Batch 的样本数量
        batch_size = images.size(0)

        # 累加 Loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 打印第一个 Batch 的形状
        if batch_idx == 0:

            print("训练 Images：", images.shape)
            print("训练 Masks：", masks.shape)
            print("训练 Prediction：", prediction.shape)

        # 打印当前 Batch Loss
        print(
            f"训练 Batch [{batch_idx + 1}/{len(train_loader)}] "
            f"Loss: {loss.item():.4f}"
        )

        # 快速测试时限制 Batch 数量
        if (
            max_batches is not None
            and batch_idx + 1 >= max_batches
        ):

            break

    # 计算平均训练 Loss
    average_loss = total_loss / total_samples

    return average_loss


# ==========================================
# 3. 验证一个 Epoch
# ==========================================

def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    max_batches=None
):

    # 设置评估模式
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    total_samples = 0

    # 验证阶段不计算梯度
    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(val_loader):

            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            prediction = model(images)

            # 计算 Loss
            loss = criterion(
                prediction,
                masks
            )

            # 检查 Loss 是否正常
            if not torch.isfinite(loss).item():

                raise RuntimeError(
                    "验证 Loss 出现 NaN 或 Inf"
                )

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

            # 当前 Batch 的样本数量
            batch_size = images.size(0)

            # 累加指标
            total_loss += loss.item() * batch_size
            total_dice += dice.item() * batch_size
            total_iou += iou.item() * batch_size

            total_samples += batch_size

            # 打印第一个 Batch 的形状
            if batch_idx == 0:

                print("验证 Images：", images.shape)
                print("验证 Prediction：", prediction.shape)

            # 打印当前 Batch 指标
            print(
                f"验证 Batch [{batch_idx + 1}/{len(val_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice.item():.4f} "
                f"IoU: {iou.item():.4f}"
            )

            # 快速测试时限制 Batch 数量
            if (
                max_batches is not None
                and batch_idx + 1 >= max_batches
            ):

                break

    # 计算平均验证指标
    average_loss = total_loss / total_samples
    average_dice = total_dice / total_samples
    average_iou = total_iou / total_samples

    return average_loss, average_dice, average_iou


# ==========================================
# 4. 主程序
# ==========================================

def main():

    # ==================================
    # 4.1 设备配置
    # ==================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("训练设备：", device)

    if device.type == "cuda":

        print(
            "显卡名称：",
            torch.cuda.get_device_name(0)
        )

    print("快速测试模式：", SMOKE_TEST)

    # ==================================
    # 4.2 创建 DataLoader
    # ==================================

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize_imagenet=True
    )

    print("\n========== 数据集检查 ==========")

    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(val_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))

    print("训练集 Batch：", len(train_loader))
    print("验证集 Batch：", len(val_loader))
    print("测试集 Batch：", len(test_loader))

    print("Batch Size：", BATCH_SIZE)

    print(
        "训练集 ImageNet 归一化：",
        train_loader.dataset.normalize_imagenet
    )

    print(
        "验证集 ImageNet 归一化：",
        val_loader.dataset.normalize_imagenet
    )

    print(
        "测试集 ImageNet 归一化：",
        test_loader.dataset.normalize_imagenet
    )

    # ==================================
    # 4.3 设置保存路径
    # ==================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    output_dir = PROJECT_ROOT / "outputs_resnet34"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    latest_path = output_dir / "latest_checkpoint.pth"
    best_path = output_dir / "best_unet.pth"
    history_path = output_dir / "training_history.csv"

    # 判断是否进行断点续训
    resume_training = (
        not SMOKE_TEST
        and latest_path.exists()
    )

    # 防止没有检查点时覆盖已有实验记录
    if (
        not SMOKE_TEST
        and not resume_training
        and (best_path.exists() or history_path.exists())
    ):

        raise RuntimeError(
            "发现已有实验记录，但缺少最新检查点，"
            "请先检查 outputs_resnet34 文件夹。"
        )

    # ==================================
    # 4.4 创建 ResNet34-U-Net
    # ==================================

    # 首次训练加载 ImageNet 预训练权重
    # 断点续训时直接恢复之前的权重
    model = ResNet34UNet(
        pretrained=not resume_training
    ).to(device)

    print("\n========== 模型创建完成 ==========")

    print("模型：ResNet34-U-Net")

    # ==================================
    # 4.5 创建损失函数
    # ==================================

    criterion = CombinedLoss()

    # ==================================
    # 4.6 创建分组学习率优化器
    # ==================================

    optimizer = torch.optim.Adam([

        # 第一组：Encoder
        {
            "params": model.encoder.parameters(),
            "lr": ENCODER_LR,
            "name": "Encoder"
        },

        # 第二组：Decoder
        {
            "params": model.decoder.parameters(),
            "lr": DECODER_LR,
            "name": "Decoder"
        },

        # 第三组：OutputLayer
        {
            "params": model.output_layer.parameters(),
            "lr": DECODER_LR,
            "name": "OutputLayer"
        }

    ])

    # ==================================
    # 4.7 检查模型参数量
    # ==================================

    encoder_params = sum(
        p.numel()
        for p in model.encoder.parameters()
    )

    decoder_params = sum(
        p.numel()
        for p in model.decoder.parameters()
    )

    output_params = sum(
        p.numel()
        for p in model.output_layer.parameters()
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n========== 模型参数检查 ==========")

    print("Encoder 参数量：", encoder_params)
    print("Decoder 参数量：", decoder_params)
    print("OutputLayer 参数量：", output_params)

    print("模型总参数量：", total_params)
    print("可训练参数量：", trainable_params)

    # ==================================
    # 4.8 初始化训练记录
    # ==================================

    start_epoch = 0
    best_val_loss = float("inf")

    last_train_loss = None
    last_val_loss = None
    last_val_dice = None
    last_val_iou = None

    checkpoint = None

    # ==================================
    # 4.9 断点续训
    # ==================================

    if resume_training:

        # 先加载到 CPU，减少额外显存占用
        checkpoint = torch.load(
            latest_path,
            map_location="cpu"
        )

        # 检查模型类型
        if checkpoint.get("model_name") != "ResNet34UNet":

            raise ValueError(
                "检查点不是本次 ResNet34-U-Net 实验保存的模型"
            )

        # 恢复模型参数
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # 恢复优化器状态
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # 恢复已经完成的 Epoch
        start_epoch = checkpoint["epoch"]

        # 恢复最近一次训练指标
        last_train_loss = checkpoint["train_loss"]
        last_val_loss = checkpoint["val_loss"]
        last_val_dice = checkpoint["val_dice"]
        last_val_iou = checkpoint["val_iou"]

        print("\n========== 断点恢复 ==========")

        print("恢复模型：", latest_path)
        print("已经完成 Epoch：", start_epoch)
        print("下一轮从 Epoch：", start_epoch + 1)

        # 恢复历史最佳验证 Loss
        if best_path.exists():

            best_checkpoint = torch.load(
                best_path,
                map_location="cpu"
            )

            best_val_loss = best_checkpoint["val_loss"]

            del best_checkpoint

        else:

            best_val_loss = last_val_loss

            torch.save(
                checkpoint,
                best_path
            )

    else:

        print("\n从 ImageNet 预训练权重开始训练。")

    # ==================================
    # 4.10 打印各参数组的学习率
    # ==================================

    print("\n========== 学习率检查 ==========")

    for group in optimizer.param_groups:

        print(
            f"{group['name']} 学习率：",
            group["lr"]
        )

    # ==================================
    # 4.11 快速测试
    # ==================================

    if SMOKE_TEST:

        print("\n========== 开始快速测试 ==========")

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()

        # 只训练两个 Batch
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=2
        )

        # 只验证两个 Batch
        val_loss, val_dice, val_iou = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=2
        )

        print("\n========== 快速测试结果 ==========")

        print("Training Loss：", train_loss)
        print("Validation Loss：", val_loss)
        print("Validation Dice：", val_dice)
        print("Validation IoU：", val_iou)

        if device.type == "cuda":

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / 1024 ** 3
            )

            print(
                f"PyTorch 峰值分配显存："
                f"{peak_memory:.2f} GiB"
            )

        print("\n快速测试通过！")
        print("本次没有保存模型和训练日志。")
        print("正式训练前请将 SMOKE_TEST 改为 False。")

        return

    # ==================================
    # 4.12 创建训练日志 CSV
    # ==================================

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou"
    ]

    if not history_path.exists():

        with open(
            history_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            # 如果从检查点恢复，但日志不存在
            if checkpoint is not None:

                writer.writerow({
                    key: checkpoint[key]
                    for key in fieldnames
                })

    # 释放加载检查点时占用的额外内存
    del checkpoint

    print("\n训练日志位置：", history_path)
    print("历史最佳验证 Loss：", best_val_loss)
    print("计划训练总 Epoch：", NUM_EPOCHS)

    # 如果已经完成训练，则不重复训练
    if start_epoch >= NUM_EPOCHS:

        print("\n模型已经完成计划训练轮数。")

        return

    # ==================================
    # 4.13 开始正式训练
    # ==================================

    for epoch in range(start_epoch, NUM_EPOCHS):

        current_epoch = epoch + 1

        print("\n" + "=" * 50)

        print(
            f"开始 Epoch "
            f"[{current_epoch}/{NUM_EPOCHS}]"
        )

        print("=" * 50)

        # ------------------------------
        # 训练
        # ------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ------------------------------
        # 验证
        # ------------------------------

        val_loss, val_dice, val_iou = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        # 保存最近一次训练指标
        last_train_loss = train_loss
        last_val_loss = val_loss
        last_val_dice = val_dice
        last_val_iou = val_iou

        # ------------------------------
        # 打印 Epoch 结果
        # ------------------------------

        print("\n========== Epoch 结果 ==========")

        print("Epoch：", current_epoch)
        print("Training Loss：", train_loss)
        print("Validation Loss：", val_loss)
        print("Validation Dice：", val_dice)
        print("Validation IoU：", val_iou)

        # ==================================
        # 4.14 保存最新检查点
        # ==================================

        current_checkpoint = {

            "model_name": "ResNet34UNet",

            "input_size": (256, 256),

            "normalize_imagenet": True,

            "batch_size": BATCH_SIZE,

            "epoch": current_epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "train_loss": train_loss,

            "val_loss": val_loss,

            "val_dice": val_dice,

            "val_iou": val_iou
        }

        torch.save(
            current_checkpoint,
            latest_path
        )

        print("\n最新模型保存成功！")

        # ==================================
        # 4.15 保存最佳模型
        # ==================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                current_checkpoint,
                best_path
            )

            print("发现更好的模型！")
            print("最佳模型已更新！")

        else:

            print(
                "验证 Loss 没有改善，"
                "保留原最佳模型。"
            )

        print(
            "当前历史最佳 Loss：",
            best_val_loss
        )

        # ==================================
        # 4.16 保存训练指标
        # ==================================

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                current_epoch,
                train_loss,
                val_loss,
                val_dice,
                val_iou
            ])

        print(
            f"Epoch {current_epoch} "
            f"的训练指标已保存！"
        )

    # ==================================
    # 4.17 训练结束
    # ==================================

    print("\n" + "=" * 50)
    print("ResNet34-U-Net 多 Epoch 训练完成！")
    print("=" * 50)

    print("已完成 Epoch：", NUM_EPOCHS)

    print("\n最近一次训练结果：")

    print("Training Loss：", last_train_loss)
    print("Validation Loss：", last_val_loss)
    print("Validation Dice：", last_val_dice)
    print("Validation IoU：", last_val_iou)

    print(
        "\n历史最佳验证 Loss：",
        best_val_loss
    )

    print("\n最新模型：", latest_path)
    print("最佳模型：", best_path)
    print("训练日志：", history_path)

    print("\n训练流程结束！")


if __name__ == "__main__":

    main()
