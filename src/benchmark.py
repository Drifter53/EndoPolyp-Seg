"""
EndoPolyp-Seg 三模型 Benchmark
--------------------------------
用途：
1. 使用 Kvasir-SEG 的真实 test split，而不是只用随机输入。
2. 分别测量 GPU 模型前向耗时（model-only）。
3. 测量真实测试集端到端耗时（DataLoader读取/预处理 + H2D + 模型前向）。
4. 统计参数量、Checkpoint 大小、MACs/FLOPs、吞吐量和 GPU 峰值显存。
5. 计算真实测试集逐图 Dice / IoU，作为推理结果检查。

运行位置：项目根目录
命令：python src/benchmark.py

注意：
- 需要 CUDA；如果 CUDA 不可用，程序会主动报错，不会悄悄改用 CPU。
- Original U-Net 的训练脚本目前未提供；本脚本按 compare_models.py 中的约定，
  尝试导入 src/model.py 中的 UNet，并读取 outputs/best_unet.pth。
  如果 model.py 或对应权重不存在，会跳过 Original U-Net，继续测试另外两个模型。
- compare_models.py 中约定 Original U-Net 不使用 ImageNet 归一化；
  ResNet34-U-Net 和 Attention U-Net 使用 ImageNet 归一化。
"""

import csv
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dataset import create_dataloaders, normalize_image
from model_resnet import ResNet34UNet
from model_attention import UNet as AttentionUNet


# ============================================================
# 1. 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_NAME = "cuda"
BATCH_SIZE = 4
NUM_WORKERS = 0
INPUT_SIZE = (256, 256)

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100

THRESHOLD = 0.5
SMOOTH = 1e-7

OUTPUT_DIR = PROJECT_ROOT / "outputs_benchmark"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_batch4.csv"
PER_IMAGE_CSV = OUTPUT_DIR / "benchmark_per_image.csv"
REPORT_TXT = OUTPUT_DIR / "benchmark_report.txt"


# ============================================================
# 2. 通用工具
# ============================================================

def safe_torch_load(checkpoint_path):
    """兼容新旧 PyTorch 的 checkpoint 加载方式。"""
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # 旧版 PyTorch 不支持 weights_only 参数
        return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint(model, checkpoint_path, expected_model_name=None,
                    expected_normalize=None):
    """兼容完整 checkpoint 和纯 state_dict，并严格加载模型参数。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint 格式错误：{checkpoint_path}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if not isinstance(state_dict, dict):
        raise ValueError(f"模型参数格式错误：{checkpoint_path}")

    actual_model_name = checkpoint.get("model_name")
    if (
        expected_model_name is not None
        and actual_model_name is not None
        and actual_model_name != expected_model_name
    ):
        raise ValueError(
            f"模型类型不匹配：期望 {expected_model_name}，"
            f"实际 {actual_model_name}"
        )

    actual_input_size = checkpoint.get("input_size")
    if actual_input_size is not None:
        if tuple(actual_input_size) != INPUT_SIZE:
            raise ValueError(
                f"Checkpoint 输入尺寸不匹配：{actual_input_size}，"
                f"当前脚本要求 {INPUT_SIZE}"
            )

    actual_normalize = checkpoint.get("normalize_imagenet")
    if (
        expected_normalize is not None
        and actual_normalize is not None
        and bool(actual_normalize) != expected_normalize
    ):
        raise ValueError(
            f"Checkpoint 归一化配置不匹配：{checkpoint_path}，"
            f"记录值={actual_normalize}，期望值={expected_normalize}"
        )

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_times(times_ms):
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": percentile(values.tolist(), 95),
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def get_checkpoint_size_mb(path):
    return path.stat().st_size / (1024 ** 2)


def profile_flops(model, device):
    """
    使用 THOP 统计单张 256x256 输入的 MACs。
    常见 FLOPs 口径按 2 * MACs 计算。
    若未安装 THOP，则返回 None，不影响其余 Benchmark。
    """
    try:
        from thop import profile
    except ImportError:
        return None, None, "未安装 thop"

    dummy = torch.randn(
        1, 3, INPUT_SIZE[0], INPUT_SIZE[1],
        device=device,
        dtype=torch.float32
    )

    try:
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs = float(macs)
        flops = 2.0 * macs
        return macs, flops, ""
    except Exception as exc:
        return None, None, f"THOP 统计失败：{exc}"


def calculate_per_image_metrics(logits, masks):
    """按每张图片分别计算 Dice 和 IoU。"""
    predictions = (torch.sigmoid(logits) >= THRESHOLD).float()
    targets = (masks >= 0.5).float()

    pred_flat = predictions.flatten(start_dim=1)
    target_flat = targets.flatten(start_dim=1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)

    dice = (
        2.0 * intersection + SMOOTH
    ) / (
        pred_sum + target_sum + SMOOTH
    )

    iou = (
        intersection + SMOOTH
    ) / (
        pred_sum + target_sum - intersection + SMOOTH
    )

    return dice, iou


def save_csv(path, records, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 3. 模型配置
# ============================================================

def build_model_specs():
    """
    Original U-Net 是可选项：
    compare_models.py 使用 from model import UNet，
    权重路径为 outputs/best_unet.pth，且不做 ImageNet 归一化。
    """
    specs = []
    skipped = []

    original_checkpoint = PROJECT_ROOT / "outputs" / "best_unet.pth"
    try:
        from model import UNet as OriginalUNet

        specs.append({
            "key": "original_unet",
            "name": "Original U-Net",
            "factory": lambda: OriginalUNet(),
            "checkpoint": original_checkpoint,
            "expected_model_name": None,
            "normalize_imagenet": False,
            "optional": True,
        })
    except Exception as exc:
        skipped.append(
            f"Original U-Net：无法导入 src/model.py 中的 UNet（{exc}）"
        )

    specs.append({
        "key": "resnet34_unet",
        "name": "ResNet34-U-Net",
        "factory": lambda: ResNet34UNet(pretrained=False),
        "checkpoint": (
            PROJECT_ROOT / "outputs_resnet34" / "best_unet.pth"
        ),
        "expected_model_name": "ResNet34UNet",
        "normalize_imagenet": True,
        "optional": False,
    })

    specs.append({
        "key": "attention_unet",
        "name": "Attention U-Net",
        "factory": lambda: AttentionUNet(),
        "checkpoint": (
            PROJECT_ROOT / "outputs_attention" / "best_attention_unet.pth"
        ),
        "expected_model_name": "AttentionUNet",
        "normalize_imagenet": True,
        "optional": False,
    })

    # 原始 U-Net 的模型代码能导入但权重不存在时，也按可选模型跳过。
    valid_specs = []
    for spec in specs:
        if not spec["checkpoint"].exists():
            message = (
                f'{spec["name"]}：找不到权重 '
                f'{spec["checkpoint"]}'
            )
            if spec["optional"]:
                skipped.append(message)
                continue
            raise FileNotFoundError(message)
        valid_specs.append(spec)

    return valid_specs, skipped


# ============================================================
# 4. GPU 模型前向 Benchmark（真实测试图片）
# ============================================================

def benchmark_model_only(model, first_images, normalize_imagenet, device):
    """
    只统计 GPU 上的模型前向耗时：
    - 使用测试集真实图片作为输入；
    - 数据读取、CPU 预处理、H2D 和归一化不计入此项；
    - 每次计时前后同步 CUDA，避免异步执行导致时间偏小。
    """
    images = first_images.to(device, non_blocking=False)

    if normalize_imagenet:
        images = normalize_image(images)

    # 预热
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = model(images)

    torch.cuda.synchronize()

    # 正式计时
    times_ms = []
    with torch.inference_mode():
        for _ in range(BENCHMARK_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(images)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

    stats = summarize_times(times_ms)
    actual_batch_size = int(images.shape[0])

    stats["mean_per_image_ms"] = stats["mean"] / actual_batch_size
    stats["throughput_img_s"] = (
        actual_batch_size * 1000.0 / stats["mean"]
    )
    return stats


# ============================================================
# 5. 真实测试集端到端 Benchmark
# ============================================================

def benchmark_test_set(model, test_loader, normalize_imagenet, device):
    """
    统计真实测试集端到端推理：
    计时范围包括：
      DataLoader 取出 batch（OpenCV 读取、resize、Tensor 转换）
      + CPU 到 GPU 传输
      + ImageNet 归一化（需要时）
      + GPU 模型前向
    Dice / IoU 在计时区间之外计算，不混入推理耗时。
    """
    model.eval()

    batch_times_ms = []
    per_image_records = []
    all_dice = []
    all_iou = []
    total_samples = 0
    sample_index = 0

    iterator = iter(test_loader)

    with torch.inference_mode():
        while True:
            torch.cuda.synchronize()
            start = time.perf_counter()

            try:
                images, masks = next(iterator)
            except StopIteration:
                break

            current_batch_size = int(images.shape[0])

            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            if normalize_imagenet:
                images = normalize_image(images)

            logits = model(images)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            batch_times_ms.append(elapsed_ms)

            dice, iou = calculate_per_image_metrics(logits, masks)

            dice_values = dice.detach().cpu().tolist()
            iou_values = iou.detach().cpu().tolist()

            for index_in_batch in range(current_batch_size):
                image_path, mask_path = (
                    test_loader.dataset.samples[
                        sample_index + index_in_batch
                    ]
                )

                per_image_records.append({
                    "image_name": image_path.name,
                    "mask_name": mask_path.name,
                    "batch_index": len(batch_times_ms),
                    "batch_size": current_batch_size,
                    "batch_latency_ms": elapsed_ms,
                    "amortized_latency_ms": (
                        elapsed_ms / current_batch_size
                    ),
                    "dice": float(dice_values[index_in_batch]),
                    "iou": float(iou_values[index_in_batch]),
                })

            all_dice.extend(dice_values)
            all_iou.extend(iou_values)

            total_samples += current_batch_size
            sample_index += current_batch_size

    if total_samples == 0:
        raise RuntimeError("测试集没有读取到任何图片。")

    stats = summarize_times(batch_times_ms)
    total_elapsed_seconds = sum(batch_times_ms) / 1000.0

    stats["throughput_img_s"] = (
        total_samples / total_elapsed_seconds
        if total_elapsed_seconds > 0 else float("nan")
    )
    stats["mean_per_image_ms"] = (
        total_elapsed_seconds * 1000.0 / total_samples
    )
    stats["test_samples"] = total_samples
    stats["mean_dice"] = float(np.mean(all_dice))
    stats["mean_iou"] = float(np.mean(all_iou))

    return stats, per_image_records


# ============================================================
# 6. 主程序
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 无法使用 CUDA。"
            "本 Benchmark 按要求只测试 cuda，不会自动切换到 CPU。"
        )

    device = torch.device(DEVICE_NAME)
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 68)
    print("EndoPolyp-Seg 三模型 Benchmark")
    print("=" * 68)
    print(f"测试设备：{device}")
    print(f"GPU：{gpu_name}")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"Batch Size：{BATCH_SIZE}")
    print(f"输入尺寸：{INPUT_SIZE[0]} x {INPUT_SIZE[1]}")
    print(f"FP32：True")
    print(f"预热次数：{WARMUP_RUNS}")
    print(f"GPU 正式计时次数：{BENCHMARK_RUNS}")

    # 读取真实测试集，统一先返回 [0,1] RGB 图像。
    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize_imagenet=False
    )

    if len(test_loader.dataset) == 0:
        raise RuntimeError("Kvasir-SEG 测试集为空，请检查 data/splits/test.txt。")

    if getattr(test_loader.dataset, "augment", False):
        raise RuntimeError("测试集不应启用数据增强。")

    if len(test_loader.dataset) < BATCH_SIZE:
        raise RuntimeError(
            f"测试集图片数量少于 Batch Size={BATCH_SIZE}，"
            "无法按当前配置进行 GPU Benchmark。"
        )

    first_images, _ = next(iter(test_loader))

    print(f"\n测试集图片数量：{len(test_loader.dataset)}")
    print(f"测试集 Batch 数量：{len(test_loader)}")
    print(f"首个 Batch 图像形状：{tuple(first_images.shape)}")

    model_specs, skipped_models = build_model_specs()

    if not model_specs:
        raise RuntimeError(
            "没有可测试的模型。请检查模型定义文件和权重路径。"
        )

    print("\n将要测试的模型：")
    for spec in model_specs:
        print(f"  - {spec['name']}")

    if skipped_models:
        print("\n跳过的模型/原因：")
        for message in skipped_models:
            print(f"  - {message}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_records = []
    all_per_image_records = []

    for model_number, spec in enumerate(model_specs, start=1):
        print("\n" + "=" * 68)
        print(f"[{model_number}/{len(model_specs)}] 测试：{spec['name']}")
        print("=" * 68)

        model = spec["factory"]().to(device)

        checkpoint = load_checkpoint(
            model,
            spec["checkpoint"],
            expected_model_name=spec["expected_model_name"],
            expected_normalize=spec["normalize_imagenet"]
        )

        model.eval()

        total_params, trainable_params = count_parameters(model)
        checkpoint_size_mb = get_checkpoint_size_mb(spec["checkpoint"])

        macs, flops, flops_note = profile_flops(model, device)

        # THOP 统计完成后再重置显存峰值，避免将复杂度分析的临时显存
        # 混入实际推理显存统计。
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        model_only_stats = benchmark_model_only(
            model,
            first_images,
            spec["normalize_imagenet"],
            device
        )

        test_stats, per_image_records = benchmark_test_set(
            model,
            test_loader,
            spec["normalize_imagenet"],
            device
        )

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        )

        for record in per_image_records:
            all_per_image_records.append({
                "model": spec["name"],
                **record
            })

        summary = {
            "model": spec["name"],
            "checkpoint": str(spec["checkpoint"].relative_to(PROJECT_ROOT)),
            "checkpoint_size_mb": checkpoint_size_mb,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameters_m": total_params / 1e6,
            "macs_per_image_g": (
                macs / 1e9 if macs is not None else ""
            ),
            "flops_per_image_g": (
                flops / 1e9 if flops is not None else ""
            ),
            "input_size": "256x256",
            "batch_size": BATCH_SIZE,
            "precision": "FP32",
            "normalize_imagenet": spec["normalize_imagenet"],
            "test_samples": test_stats["test_samples"],

            "gpu_mean_batch_ms": model_only_stats["mean"],
            "gpu_median_batch_ms": model_only_stats["median"],
            "gpu_p95_batch_ms": model_only_stats["p95"],
            "gpu_mean_per_image_ms": model_only_stats["mean_per_image_ms"],
            "gpu_throughput_img_s": model_only_stats["throughput_img_s"],

            "e2e_mean_batch_ms": test_stats["mean"],
            "e2e_median_batch_ms": test_stats["median"],
            "e2e_p95_batch_ms": test_stats["p95"],
            "e2e_mean_per_image_ms": test_stats["mean_per_image_ms"],
            "e2e_throughput_img_s": test_stats["throughput_img_s"],

            "test_mean_dice": test_stats["mean_dice"],
            "test_mean_iou": test_stats["mean_iou"],

            "peak_memory_allocated_gb": peak_allocated_gb,
            "peak_memory_reserved_gb": peak_reserved_gb,
            "thop_note": flops_note,
        }
        summary_records.append(summary)

        print(f"参数量：{total_params / 1e6:.2f} M")
        print(f"Checkpoint：{checkpoint_size_mb:.2f} MB")

        if macs is not None:
            print(f"MACs：{macs / 1e9:.2f} G / image")
            print(f"FLOPs：{flops / 1e9:.2f} G / image")
        else:
            print(f"复杂度统计：{flops_note}")

        print("\nGPU 模型前向（真实测试图片，排除数据读取和传输）：")
        print(f"  平均 Batch 延迟：{model_only_stats['mean']:.3f} ms")
        print(f"  中位 Batch 延迟：{model_only_stats['median']:.3f} ms")
        print(f"  P95 Batch 延迟：{model_only_stats['p95']:.3f} ms")
        print(f"  吞吐量：{model_only_stats['throughput_img_s']:.2f} images/s")

        print("\n真实测试集端到端（包含 DataLoader、传输、归一化和前向）：")
        print(f"  平均 Batch 延迟：{test_stats['mean']:.3f} ms")
        print(f"  中位 Batch 延迟：{test_stats['median']:.3f} ms")
        print(f"  P95 Batch 延迟：{test_stats['p95']:.3f} ms")
        print(f"  吞吐量：{test_stats['throughput_img_s']:.2f} images/s")
        print(f"  Test Dice：{test_stats['mean_dice']:.4f}")
        print(f"  Test IoU：{test_stats['mean_iou']:.4f}")

        print("\nGPU 峰值显存：")
        print(f"  Allocated：{peak_allocated_gb:.3f} GiB")
        print(f"  Reserved：{peak_reserved_gb:.3f} GiB")

        if "epoch" in checkpoint:
            print(f"\nCheckpoint Epoch：{checkpoint['epoch']}")

        # 每个模型完成后释放显存，避免三模型同时占用 GPU。
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    summary_fields = [
        "model",
        "checkpoint",
        "checkpoint_size_mb",
        "parameters",
        "trainable_parameters",
        "parameters_m",
        "macs_per_image_g",
        "flops_per_image_g",
        "input_size",
        "batch_size",
        "precision",
        "normalize_imagenet",
        "test_samples",
        "gpu_mean_batch_ms",
        "gpu_median_batch_ms",
        "gpu_p95_batch_ms",
        "gpu_mean_per_image_ms",
        "gpu_throughput_img_s",
        "e2e_mean_batch_ms",
        "e2e_median_batch_ms",
        "e2e_p95_batch_ms",
        "e2e_mean_per_image_ms",
        "e2e_throughput_img_s",
        "test_mean_dice",
        "test_mean_iou",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "thop_note",
    ]

    per_image_fields = [
        "model",
        "image_name",
        "mask_name",
        "batch_index",
        "batch_size",
        "batch_latency_ms",
        "amortized_latency_ms",
        "dice",
        "iou",
    ]

    save_csv(SUMMARY_CSV, summary_records, summary_fields)
    save_csv(PER_IMAGE_CSV, all_per_image_records, per_image_fields)

    # 文字报告
    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("EndoPolyp-Seg 三模型 Benchmark 报告\n")
        file.write("=" * 68 + "\n")
        file.write(f"GPU：{gpu_name}\n")
        file.write(f"PyTorch：{torch.__version__}\n")
        file.write(f"CUDA：{torch.version.cuda}\n")
        file.write(f"输入尺寸：{INPUT_SIZE[0]}x{INPUT_SIZE[1]}\n")
        file.write(f"Batch Size：{BATCH_SIZE}\n")
        file.write("精度：FP32\n")
        file.write(f"预热次数：{WARMUP_RUNS}\n")
        file.write(f"GPU 正式计时次数：{BENCHMARK_RUNS}\n")
        file.write(f"测试集图片数量：{len(test_loader.dataset)}\n\n")

        file.write("计时口径说明：\n")
        file.write(
            "1. GPU 模型前向：使用真实测试图片，排除数据读取、CPU预处理和H2D传输。\n"
        )
        file.write(
            "2. 端到端：包括DataLoader取数、CPU预处理、H2D传输、归一化和模型前向。\n"
        )
        file.write(
            "3. FLOPs按2×MACs计算；THOP未安装或统计失败时会留空并写明原因。\n\n"
        )

        if skipped_models:
            file.write("跳过的模型/原因：\n")
            for message in skipped_models:
                file.write(f"- {message}\n")
            file.write("\n")

        for result in summary_records:
            file.write("-" * 68 + "\n")
            file.write(f"模型：{result['model']}\n")
            file.write(f"Checkpoint：{result['checkpoint']}\n")
            file.write(
                f"参数量：{result['parameters_m']:.3f} M\n"
            )
            file.write(
                f"Checkpoint大小：{result['checkpoint_size_mb']:.2f} MB\n"
            )
            if result["macs_per_image_g"] != "":
                file.write(
                    f"MACs：{result['macs_per_image_g']:.3f} G / image\n"
                )
                file.write(
                    f"FLOPs：{result['flops_per_image_g']:.3f} G / image\n"
                )
            else:
                file.write(f"复杂度统计：{result['thop_note']}\n")

            file.write("\nGPU模型前向：\n")
            file.write(
                f"平均延迟：{result['gpu_mean_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"中位延迟：{result['gpu_median_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"P95延迟：{result['gpu_p95_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"吞吐量：{result['gpu_throughput_img_s']:.2f} images/s\n"
            )

            file.write("\n真实测试集端到端：\n")
            file.write(
                f"平均延迟：{result['e2e_mean_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"中位延迟：{result['e2e_median_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"P95延迟：{result['e2e_p95_batch_ms']:.3f} ms / batch\n"
            )
            file.write(
                f"吞吐量：{result['e2e_throughput_img_s']:.2f} images/s\n"
            )
            file.write(f"Test Dice：{result['test_mean_dice']:.4f}\n")
            file.write(f"Test IoU：{result['test_mean_iou']:.4f}\n")
            file.write(
                f"峰值Allocated显存：{result['peak_memory_allocated_gb']:.3f} GiB\n"
            )
            file.write(
                f"峰值Reserved显存：{result['peak_memory_reserved_gb']:.3f} GiB\n"
            )
            file.write("\n")

    print("\n" + "=" * 68)
    print("Benchmark 全部完成！")
    print("=" * 68)
    print(f"汇总 CSV：{SUMMARY_CSV}")
    print(f"逐图 CSV：{PER_IMAGE_CSV}")
    print(f"文字报告：{REPORT_TXT}")

    if any(result["thop_note"] for result in summary_records):
        print("\n提示：如需 MACs/FLOPs，请在当前环境安装 THOP：")
        print("pip install thop")


if __name__ == "__main__":
    main()
