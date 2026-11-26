
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ==================================
# 1. 设置项目路径
# ==================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

output_dir = PROJECT_ROOT / "outputs_resnet34"

history_path = output_dir / "training_history.csv"


# ==================================
# 2. 读取训练日志
# ==================================

epochs = []

train_losses = []
val_losses = []

val_dices = []
val_ious = []

with open(
    history_path,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        epochs.append(int(row["epoch"]))

        train_losses.append(
            float(row["train_loss"])
        )

        val_losses.append(
            float(row["val_loss"])
        )

        val_dices.append(
            float(row["val_dice"])
        )

        val_ious.append(
            float(row["val_iou"])
        )


# ==================================
# 3. 绘制 Loss 曲线
# ==================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("U-Net Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = output_dir / "loss_curve.png"

plt.savefig(
    loss_path,
    dpi=150
)

plt.close()


# ==================================
# 4. 绘制 Dice 和 IoU 曲线
# ==================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    val_dices,
    marker="o",
    label="Validation Dice"
)

plt.plot(
    epochs,
    val_ious,
    marker="o",
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("Score")

plt.title("U-Net Validation Dice and IoU")

plt.ylim(0, 1)

plt.legend()
plt.grid(True)

plt.tight_layout()

metrics_path = output_dir / "metrics_curve.png"

plt.savefig(
    metrics_path,
    dpi=150
)

plt.close()


# ==================================
# 5. 输出结果
# ==================================

print("训练记录数量：", len(epochs))

print("Loss曲线保存成功：", loss_path)

print("Dice和IoU曲线保存成功：", metrics_path)
