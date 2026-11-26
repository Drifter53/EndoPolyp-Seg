"""
EndoPolyp-Seg Benchmark V3
==========================

在 V2 基础上修正稳态计时口径，并针对不完整尾批进行 cuDNN 预热。

核心原则：
1. 三个模型、checkpoint、测试集、预处理约定均沿用 V2。
2. 保留完整 test split 的逐图 Dice / IoU（150 张全部参与）。
3. 正式测速前，分别预热 batch=4 和尾批 batch=2 的输入形状。
4. 稳态速度只统计 batch=4 且排除正式测试首批的记录。
5. 最后一批 batch=2 单独记录，不混入 batch=4 稳态统计。
6. 同时输出完整测试遍历吞吐量，方便了解实际全流程表现。
7. 不修改任何模型权重或训练代码。

运行：
    python src/benchmark_v3.py

依赖：
    pip install thop

注意：
- 仍沿用 V2 的模型导入路径、checkpoint 路径和 normalize_image 函数。
- Original U-Net 仍按 V2 设置不做 ImageNet normalization。
  该预处理约定尚未通过原始训练脚本独立核实，不要在没有核实前擅自改变。
- cuDNN benchmark 的形状预热是针对 V2 尾批异常的诊断性修正；
  如果 V3 仍出现异常，需要结合逐 Batch 数据继续定位，不能仅凭一次结果断言原因。
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from dataset import create_dataloaders, normalize_image
from model_resnet import ResNet34UNet
from model_attention import UNet as AttentionUNet


# ============================================================
# 1. Benchmark 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda")

BATCH_SIZE = 4
NUM_WORKERS = 0
INPUT_SIZE = (256, 256)

MODEL_ONLY_WARMUP_RUNS = 20
MODEL_ONLY_MEASURE_RUNS = 100

E2E_WARMUP_BATCHES = 3
CUDNN_SHAPE_WARMUP_RUNS = 5

THRESHOLD = 0.5
SMOOTH = 1e-7

OUTPUT_DIR = PROJECT_ROOT / "outputs_benchmark_v3"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_v3_batch4.csv"
PER_IMAGE_CSV = OUTPUT_DIR / "benchmark_v3_per_image.csv"
PER_BATCH_CSV = OUTPUT_DIR / "benchmark_v3_per_batch.csv"
REPORT_TXT = OUTPUT_DIR / "benchmark_v3_report.txt"


# ============================================================
# 2. 通用函数
# ============================================================

def safe_torch_load(checkpoint_path):
    """兼容新旧 PyTorch 的 checkpoint 加载。"""
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint(
    model,
    checkpoint_path,
    expected_model_name=None,
    expected_normalize=None
):
    """加载完整 checkpoint 或纯 state_dict，并严格检查参数。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint 格式错误：{checkpoint_path}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"无法从 checkpoint 读取 state_dict：{checkpoint_path}")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    actual_model_name = checkpoint.get("model_name")
    if (
        expected_model_name is not None
        and actual_model_name is not None
        and actual_model_name != expected_model_name
    ):
        raise ValueError(
            f"模型类型不匹配：期望 {expected_model_name}，"
            f"checkpoint 记录为 {actual_model_name}"
        )

    actual_input_size = checkpoint.get("input_size")
    if actual_input_size is not None:
        if tuple(actual_input_size) != INPUT_SIZE:
            raise ValueError(
                f"Checkpoint 输入尺寸为 {actual_input_size}，"
                f"当前 Benchmark 要求 {INPUT_SIZE}"
            )

    actual_normalize = checkpoint.get("normalize_imagenet")
    if (
        expected_normalize is not None
        and actual_normalize is not None
        and bool(actual_normalize) != expected_normalize
    ):
        raise ValueError(
            f"归一化配置不匹配：checkpoint={actual_normalize}，"
            f"当前配置={expected_normalize}，文件={checkpoint_path}"
        )

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def summarize_times(values):
    """返回一组毫秒数据的常用统计量。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def checkpoint_size_mb(path):
    return path.stat().st_size / (1024 ** 2)


def profile_complexity(model, device):
    """THOP 统计单张 256x256 输入的 MACs，FLOPs 按 2×MACs 估算。"""
    try:
        from thop import profile
    except ImportError:
        return None, None, "未安装 thop；运行 pip install thop"

    dummy = torch.randn(
        1, 3, INPUT_SIZE[0], INPUT_SIZE[1],
        device=device,
        dtype=torch.float32
    )

    try:
        model.eval()
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs = float(macs)
        return macs, 2.0 * macs, ""
    except Exception as exc:
        return None, None, f"THOP 统计失败：{exc}"


def calculate_per_image_metrics(logits, masks_cpu):
    """在计时结束后计算逐图 Dice / IoU，mask 保留在 CPU。"""
    predictions = (
        torch.sigmoid(logits.detach().float().cpu()) >= THRESHOLD
    ).float()
    targets = (masks_cpu >= 0.5).float()

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

    return dice.tolist(), iou.tolist()


def save_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def get_sample_names(dataset, start_index, batch_size):
    """按 Dataset.samples 的顺序获取图片和 mask 文件名。"""
    names = []
    samples = getattr(dataset, "samples", None)

    for offset in range(batch_size):
        index = start_index + offset

        if samples is not None and index < len(samples):
            sample = samples[index]
            try:
                image_path, mask_path = sample[0], sample[1]
                image_name = Path(str(image_path)).name
                mask_name = Path(str(mask_path)).name
            except Exception:
                image_name = f"test_image_{index:04d}"
                mask_name = f"test_mask_{index:04d}"
        else:
            image_name = f"test_image_{index:04d}"
            mask_name = f"test_mask_{index:04d}"

        names.append((image_name, mask_name))

    return names


def cuda_event_pair():
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True)
    )


# ============================================================
# 3. 模型配置：沿用 V2
# ============================================================

def build_model_specs():
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
            "Original U-Net：无法导入 src/model.py 中的 UNet，"
            f"原因：{exc}"
        )

    specs.append({
        "key": "resnet34_unet",
        "name": "ResNet34-U-Net",
        "factory": lambda: ResNet34UNet(pretrained=False),
        "checkpoint": PROJECT_ROOT / "outputs_resnet34" / "best_unet.pth",
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

    valid_specs = []
    for spec in specs:
        if not spec["checkpoint"].exists():
            message = (
                f"{spec['name']}：找不到权重 {spec['checkpoint']}"
            )
            if spec["optional"]:
                skipped.append(message)
                continue
            raise FileNotFoundError(message)
        valid_specs.append(spec)

    return valid_specs, skipped


# ============================================================
# 4. 针对不同 Batch Shape 预热 cuDNN
# ============================================================

def prewarm_cudnn_shapes(
    model,
    normalize_imagenet,
    device,
    batch_sizes
):
    """
    在正式计时前预热所有实际会出现的 batch shape。

    例如 150 张图片、batch size=4：
      - 完整 batch：4
      - 最后一批：2

    cudnn.benchmark=True 会根据输入 shape 选择卷积算法。
    这里先运行 batch=4 和 batch=2，避免尾批第一次出现时才搜索算法。
    """
    unique_batch_sizes = sorted({
        int(size) for size in batch_sizes if int(size) > 0
    })

    model.eval()
    warmed = []

    with torch.inference_mode():
        for batch_size in unique_batch_sizes:
            dummy = torch.rand(
                batch_size,
                3,
                INPUT_SIZE[0],
                INPUT_SIZE[1],
                device=device,
                dtype=torch.float32
            )

            if normalize_imagenet:
                dummy = normalize_image(dummy)

            for _ in range(CUDNN_SHAPE_WARMUP_RUNS):
                _ = model(dummy)

            torch.cuda.synchronize()
            warmed.append(batch_size)
            del dummy

    return warmed


# ============================================================
# 5. GPU-only 前向测速
# ============================================================

def benchmark_model_only(
    model,
    first_images_cpu,
    normalize_imagenet,
    device
):
    """只计 GPU 模型 forward；H2D 和 normalization 不计入。"""
    images = first_images_cpu.to(device, non_blocking=False)

    if normalize_imagenet:
        images = normalize_image(images)

    model.eval()
    start_event, end_event = cuda_event_pair()

    with torch.inference_mode():
        for _ in range(MODEL_ONLY_WARMUP_RUNS):
            _ = model(images)

        torch.cuda.synchronize()

        times_ms = []
        for _ in range(MODEL_ONLY_MEASURE_RUNS):
            start_event.record()
            _ = model(images)
            end_event.record()
            end_event.synchronize()
            times_ms.append(float(start_event.elapsed_time(end_event)))

    stats = summarize_times(times_ms)
    batch_size = int(images.shape[0])
    stats["batch_size"] = batch_size
    stats["mean_per_image_ms"] = stats["mean"] / batch_size
    stats["throughput_img_s"] = (
        batch_size * 1000.0 / stats["mean"]
        if stats["mean"] > 0 else float("nan")
    )
    return stats


# ============================================================
# 6. 端到端预热
# ============================================================

def warmup_end_to_end(model, test_loader, normalize_imagenet, device):
    """每个模型独立预热 DataLoader、H2D、归一化和 forward。"""
    model.eval()
    iterator = iter(test_loader)
    warmed_batches = 0

    with torch.inference_mode():
        for _ in range(min(E2E_WARMUP_BATCHES, len(test_loader))):
            try:
                images, _masks = next(iterator)
            except StopIteration:
                break

            images = images.to(device, non_blocking=False)
            if normalize_imagenet:
                images = normalize_image(images)

            _ = model(images)
            warmed_batches += 1

    torch.cuda.synchronize()
    del iterator
    return warmed_batches


# ============================================================
# 7. 端到端正式测试
# ============================================================

def benchmark_end_to_end(
    model,
    test_loader,
    normalize_imagenet,
    device
):
    """
    完整遍历 test split 一次。

    每个 batch 的 E2E wall time 包含：
      DataLoader next + H2D + GPU normalization + GPU forward + CUDA sync

    统计口径：
      - first_batch：正式遍历首批，单独记录。
      - steady_full_batch：batch size=4 且不是首批，纳入稳态统计。
      - tail_partial_batch：最后不足 4 张的尾批，单独记录。
      - 所有 batch 均参与逐图 Dice / IoU。
    """
    model.eval()

    iterator_start = time.perf_counter()
    iterator = iter(test_loader)
    iterator_init_ms = (time.perf_counter() - iterator_start) * 1000.0

    all_batch_records = []
    per_image_records = []
    all_dice = []
    all_iou = []

    total_samples = 0
    sample_index = 0
    batch_index = 0

    with torch.inference_mode():
        while True:
            torch.cuda.synchronize()
            e2e_start = time.perf_counter()

            loader_start = time.perf_counter()
            try:
                images_cpu, masks_cpu = next(iterator)
            except StopIteration:
                break

            loader_wait_ms = (
                time.perf_counter() - loader_start
            ) * 1000.0

            current_batch_size = int(images_cpu.shape[0])

            h2d_start, h2d_end = cuda_event_pair()
            h2d_start.record()
            images_gpu = images_cpu.to(device, non_blocking=False)
            h2d_end.record()

            norm_start, norm_end = cuda_event_pair()
            norm_start.record()
            if normalize_imagenet:
                images_gpu = normalize_image(images_gpu)
            norm_end.record()

            forward_start, forward_end = cuda_event_pair()
            forward_start.record()
            logits = model(images_gpu)
            forward_end.record()

            torch.cuda.synchronize()
            e2e_ms = (time.perf_counter() - e2e_start) * 1000.0

            h2d_ms = float(h2d_start.elapsed_time(h2d_end))
            normalize_ms = float(norm_start.elapsed_time(norm_end))
            forward_ms = float(forward_start.elapsed_time(forward_end))

            # 计时结束后再算指标。
            dice_values, iou_values = calculate_per_image_metrics(
                logits,
                masks_cpu
            )

            names = get_sample_names(
                test_loader.dataset,
                sample_index,
                current_batch_size
            )

            batch_index += 1
            is_first_batch = (batch_index == 1)
            is_partial_batch = (current_batch_size < BATCH_SIZE)

            if is_first_batch:
                batch_role = "first_batch"
            elif is_partial_batch:
                batch_role = "tail_partial_batch"
            else:
                batch_role = "steady_full_batch"

            batch_record = {
                "batch_index": batch_index,
                "batch_size": current_batch_size,
                "batch_role": batch_role,
                "e2e_latency_ms": e2e_ms,
                "loader_wait_ms": loader_wait_ms,
                "h2d_ms": h2d_ms,
                "gpu_normalize_ms": normalize_ms,
                "gpu_forward_ms": forward_ms,
                "amortized_e2e_ms_per_image": (
                    e2e_ms / current_batch_size
                ),
            }
            all_batch_records.append(batch_record)

            for item_index in range(current_batch_size):
                image_name, mask_name = names[item_index]
                per_image_records.append({
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "batch_index": batch_index,
                    "batch_size": current_batch_size,
                    "dice": float(dice_values[item_index]),
                    "iou": float(iou_values[item_index]),
                })

            all_dice.extend(dice_values)
            all_iou.extend(iou_values)

            total_samples += current_batch_size
            sample_index += current_batch_size

            del images_gpu, logits

    del iterator

    if total_samples == 0:
        raise RuntimeError("测试集没有读取到任何图片。")

    first_records = [
        record for record in all_batch_records
        if record["batch_role"] == "first_batch"
    ]
    steady_records = [
        record for record in all_batch_records
        if record["batch_role"] == "steady_full_batch"
    ]
    tail_records = [
        record for record in all_batch_records
        if record["batch_role"] == "tail_partial_batch"
    ]

    def component_stats(records, key):
        return summarize_times([record[key] for record in records])

    steady_times = [record["e2e_latency_ms"] for record in steady_records]
    steady_images = sum(record["batch_size"] for record in steady_records)
    steady_total_ms = sum(steady_times)

    full_pass_times = [
        record["e2e_latency_ms"] for record in all_batch_records
    ]
    full_pass_total_ms = sum(full_pass_times)

    first_record = first_records[0] if first_records else {}
    tail_record = tail_records[-1] if tail_records else {}

    result = {
        "iterator_init_ms": iterator_init_ms,

        "first_batch_e2e_ms": first_record.get("e2e_latency_ms", float("nan")),
        "first_batch_forward_ms": first_record.get("gpu_forward_ms", float("nan")),

        "tail_batch_size": tail_record.get("batch_size", 0),
        "tail_batch_e2e_ms": tail_record.get("e2e_latency_ms", float("nan")),
        "tail_batch_forward_ms": tail_record.get("gpu_forward_ms", float("nan")),

        "steady_batch_count": len(steady_records),
        "steady_image_count": steady_images,
        "test_samples": total_samples,
        "test_batches": len(all_batch_records),

        "steady_e2e": summarize_times(steady_times),
        "steady_loader_wait": component_stats(steady_records, "loader_wait_ms"),
        "steady_h2d": component_stats(steady_records, "h2d_ms"),
        "steady_normalize": component_stats(
            steady_records, "gpu_normalize_ms"
        ),
        "steady_forward": component_stats(steady_records, "gpu_forward_ms"),

        "steady_throughput_img_s": (
            steady_images * 1000.0 / steady_total_ms
            if steady_total_ms > 0 else float("nan")
        ),
        "steady_amortized_ms_per_image": (
            steady_total_ms / steady_images
            if steady_images > 0 else float("nan")
        ),
        "full_pass_throughput_img_s": (
            total_samples * 1000.0 / full_pass_total_ms
            if full_pass_total_ms > 0 else float("nan")
        ),

        "mean_dice": float(np.mean(all_dice)),
        "mean_iou": float(np.mean(all_iou)),
    }

    return result, per_image_records, all_batch_records


# ============================================================
# 8. 主程序
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。本脚本要求使用 CUDA，不会自动切换到 CPU。\n"
            "请使用项目的 .venv-gpu311 Python 解释器运行。"
        )

    # 保留 V2 的 cuDNN benchmark 设置，但提前预热每种实际 batch shape。
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    device = DEVICE
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 72)
    print("EndoPolyp-Seg Benchmark V3")
    print("=" * 72)
    print(f"GPU：{gpu_name}")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"Batch Size：{BATCH_SIZE}")
    print(f"输入尺寸：{INPUT_SIZE[0]} x {INPUT_SIZE[1]}")
    print("精度：FP32")
    print(f"cuDNN benchmark：{torch.backends.cudnn.benchmark}")
    print(f"cuDNN shape 预热：每种 shape {CUDNN_SHAPE_WARMUP_RUNS} 次")
    print(f"GPU-only 预热：{MODEL_ONLY_WARMUP_RUNS} 次")
    print(f"GPU-only 正式计时：{MODEL_ONLY_MEASURE_RUNS} 次")
    print(f"端到端预热：每模型最多 {E2E_WARMUP_BATCHES} 个 batch")
    print("稳态统计：只统计完整 Batch=4，排除首批")
    print("尾批：单独记录，但仍参与完整测试集 Dice / IoU")
    print(f"输出目录：{OUTPUT_DIR}")

    _train_loader, _val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize_imagenet=False
    )

    dataset_size = len(test_loader.dataset)
    if dataset_size == 0:
        raise RuntimeError("测试集为空，请检查 Kvasir-SEG test split。")

    if not isinstance(test_loader.sampler, SequentialSampler):
        raise RuntimeError(
            "当前 test_loader 不是顺序采样。请确认测试集 DataLoader 使用 shuffle=False。"
        )

    if getattr(test_loader.dataset, "augment", False):
        raise RuntimeError("检测到测试集启用了数据增强，请关闭 test augmentation。")

    if getattr(test_loader, "drop_last", False):
        raise RuntimeError("test_loader.drop_last=True，会丢弃测试集尾部图片。")

    if getattr(test_loader, "batch_size", None) != BATCH_SIZE:
        raise RuntimeError(
            f"test_loader.batch_size={test_loader.batch_size}，"
            f"预期为 {BATCH_SIZE}"
        )

    first_images, _first_masks = next(iter(test_loader))
    if tuple(first_images.shape[-2:]) != INPUT_SIZE:
        raise RuntimeError(
            f"Dataset 实际输出尺寸为 {tuple(first_images.shape[-2:])}，"
            f"预期为 {INPUT_SIZE}"
        )

    remainder = dataset_size % BATCH_SIZE
    tail_batch_size = remainder if remainder else BATCH_SIZE
    actual_batch_shapes = sorted({BATCH_SIZE, tail_batch_size})

    print(f"\n测试集图片数：{dataset_size}")
    print(f"测试集 batch 数：{len(test_loader)}")
    print(f"首批输入形状：{tuple(first_images.shape)}")
    print(f"需要预热的 Batch Size：{actual_batch_shapes}")
    print(f"最后一批图片数：{tail_batch_size}")

    model_specs, skipped_models = build_model_specs()

    if not model_specs:
        raise RuntimeError("没有可用模型，请检查模型代码和 checkpoint。")

    if skipped_models:
        print("\n跳过的模型/原因：")
        for message in skipped_models:
            print(" -", message)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_records = []
    all_per_image_records = []
    all_per_batch_records = []

    for model_number, spec in enumerate(model_specs, start=1):
        print("\n" + "=" * 72)
        print(f"[{model_number}/{len(model_specs)}] {spec['name']}")
        print("=" * 72)

        model = None
        try:
            model = spec["factory"]().to(device)
            checkpoint = load_checkpoint(
                model,
                spec["checkpoint"],
                expected_model_name=spec["expected_model_name"],
                expected_normalize=spec["normalize_imagenet"]
            )
            model.eval()

            total_params, trainable_params = count_parameters(model)
            ckpt_mb = checkpoint_size_mb(spec["checkpoint"])
            macs, flops, complexity_note = profile_complexity(model, device)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            print(f"参数量：{total_params / 1e6:.2f} M")
            print(f"Checkpoint：{ckpt_mb:.2f} MB")
            if macs is not None:
                print(f"MACs：{macs / 1e9:.2f} G / image")
                print(f"FLOPs：{flops / 1e9:.2f} G / image")
            else:
                print(f"复杂度统计：{complexity_note}")

            # 关键修正：正式计时前，预热完整 batch 和尾批两种 shape。
            print("\nA. cuDNN 输入形状预热")
            warmed_shapes = prewarm_cudnn_shapes(
                model,
                spec["normalize_imagenet"],
                device,
                actual_batch_shapes
            )
            print(
                f"  已预热 Batch Size：{warmed_shapes}，"
                f"每种 shape {CUDNN_SHAPE_WARMUP_RUNS} 次"
            )

            print("\nB. GPU model-only 测速（Batch=4）")
            model_only = benchmark_model_only(
                model,
                first_images,
                spec["normalize_imagenet"],
                device
            )
            print(
                f"  Mean / Median / P95："
                f"{model_only['mean']:.3f} / "
                f"{model_only['median']:.3f} / "
                f"{model_only['p95']:.3f} ms / batch"
            )
            print(
                f"  Min / Max / Std："
                f"{model_only['min']:.3f} / "
                f"{model_only['max']:.3f} / "
                f"{model_only['std']:.3f} ms"
            )
            print(
                f"  吞吐量：{model_only['throughput_img_s']:.2f} images/s"
            )

            print("\nC. 端到端预热")
            warmed_batches = warmup_end_to_end(
                model,
                test_loader,
                spec["normalize_imagenet"],
                device
            )
            print(f"  已完成 {warmed_batches} 个 warm-up batch")

            print("\nD. 完整测试集端到端测试")
            e2e, per_image_records, batch_records = benchmark_end_to_end(
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

            steady = e2e["steady_e2e"]

            print(
                f"  正式遍历首批：{e2e['first_batch_e2e_ms']:.3f} ms "
                f"（单独记录）"
            )
            print(
                f"  尾批 Batch={e2e['tail_batch_size']}："
                f"{e2e['tail_batch_e2e_ms']:.3f} ms "
                f"（单独记录）"
            )
            print(
                f"  稳态完整 Batch=4 Mean / Median / P95："
                f"{steady['mean']:.3f} / "
                f"{steady['median']:.3f} / "
                f"{steady['p95']:.3f} ms"
            )
            print(
                f"  稳态 Min / Max / Std："
                f"{steady['min']:.3f} / "
                f"{steady['max']:.3f} / "
                f"{steady['std']:.3f} ms"
            )
            print(
                f"  稳态吞吐量："
                f"{e2e['steady_throughput_img_s']:.2f} images/s"
            )
            print(
                f"  完整遍历吞吐量（含首批和尾批）："
                f"{e2e['full_pass_throughput_img_s']:.2f} images/s"
            )

            print("\n  稳态分项耗时（每 Batch 平均）：")
            print(
                f"    DataLoader + CPU读取/预处理："
                f"{e2e['steady_loader_wait']['mean']:.3f} ms"
            )
            print(f"    H2D：{e2e['steady_h2d']['mean']:.3f} ms")
            print(
                f"    GPU normalization："
                f"{e2e['steady_normalize']['mean']:.3f} ms"
            )
            print(
                f"    GPU forward："
                f"{e2e['steady_forward']['mean']:.3f} ms"
            )

            print(f"\n  完整测试集图片数：{e2e['test_samples']}")
            print(f"  完整测试集 Dice：{e2e['mean_dice']:.4f}")
            print(f"  完整测试集 IoU：{e2e['mean_iou']:.4f}")
            print(
                f"  GPU 峰值显存 Allocated / Reserved："
                f"{peak_allocated_gb:.3f} / {peak_reserved_gb:.3f} GiB"
            )

            for record in per_image_records:
                all_per_image_records.append({
                    "model": spec["name"],
                    **record
                })

            for record in batch_records:
                all_per_batch_records.append({
                    "model": spec["name"],
                    **record
                })

            summary = {
                "model": spec["name"],
                "checkpoint": str(
                    spec["checkpoint"].relative_to(PROJECT_ROOT)
                ),
                "checkpoint_size_mb": ckpt_mb,
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
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_shape_warmup_runs": CUDNN_SHAPE_WARMUP_RUNS,
                "warmed_batch_sizes": str(warmed_shapes),
                "normalize_imagenet": spec["normalize_imagenet"],
                "test_samples": e2e["test_samples"],
                "test_batches": e2e["test_batches"],

                "gpu_mean_batch_ms": model_only["mean"],
                "gpu_median_batch_ms": model_only["median"],
                "gpu_p95_batch_ms": model_only["p95"],
                "gpu_min_batch_ms": model_only["min"],
                "gpu_max_batch_ms": model_only["max"],
                "gpu_std_batch_ms": model_only["std"],
                "gpu_mean_per_image_ms": model_only["mean_per_image_ms"],
                "gpu_throughput_img_s": model_only["throughput_img_s"],

                "iterator_init_ms": e2e["iterator_init_ms"],
                "first_batch_e2e_ms": e2e["first_batch_e2e_ms"],
                "tail_batch_size": e2e["tail_batch_size"],
                "tail_batch_e2e_ms": e2e["tail_batch_e2e_ms"],
                "tail_batch_forward_ms": e2e["tail_batch_forward_ms"],

                "steady_full_batch_count": e2e["steady_batch_count"],
                "steady_full_image_count": e2e["steady_image_count"],
                "e2e_steady_mean_batch_ms": steady["mean"],
                "e2e_steady_median_batch_ms": steady["median"],
                "e2e_steady_p95_batch_ms": steady["p95"],
                "e2e_steady_min_batch_ms": steady["min"],
                "e2e_steady_max_batch_ms": steady["max"],
                "e2e_steady_std_batch_ms": steady["std"],
                "e2e_steady_mean_per_image_ms": (
                    e2e["steady_amortized_ms_per_image"]
                ),
                "e2e_steady_throughput_img_s": (
                    e2e["steady_throughput_img_s"]
                ),
                "e2e_full_pass_throughput_img_s": (
                    e2e["full_pass_throughput_img_s"]
                ),

                "loader_wait_mean_ms": e2e["steady_loader_wait"]["mean"],
                "h2d_mean_ms": e2e["steady_h2d"]["mean"],
                "gpu_normalize_mean_ms": e2e["steady_normalize"]["mean"],
                "gpu_forward_mean_ms": e2e["steady_forward"]["mean"],

                "test_mean_dice": e2e["mean_dice"],
                "test_mean_iou": e2e["mean_iou"],
                "peak_memory_allocated_gb": peak_allocated_gb,
                "peak_memory_reserved_gb": peak_reserved_gb,
                "complexity_note": complexity_note,
                "checkpoint_epoch": checkpoint.get("epoch", ""),
                "e2e_warmup_batches": warmed_batches,
            }
            summary_records.append(summary)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"{spec['name']} 发生 CUDA OOM。"
                    "请先关闭 Gradio 等占用显存的程序，再重试；"
                    "如果仍然 OOM，可将 BATCH_SIZE 从 4 降到 2。"
                ) from exc
            raise
        finally:
            if model is not None:
                del model
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ========================================================
    # 9. 输出 CSV
    # ========================================================

    if not summary_records:
        raise RuntimeError("没有模型成功完成 Benchmark。")

    summary_fields = list(summary_records[0].keys())

    per_image_fields = [
        "model", "image_name", "mask_name",
        "batch_index", "batch_size", "dice", "iou"
    ]

    per_batch_fields = [
        "model", "batch_index", "batch_size", "batch_role",
        "e2e_latency_ms", "loader_wait_ms", "h2d_ms",
        "gpu_normalize_ms", "gpu_forward_ms",
        "amortized_e2e_ms_per_image"
    ]

    save_csv(SUMMARY_CSV, summary_records, summary_fields)
    save_csv(PER_IMAGE_CSV, all_per_image_records, per_image_fields)
    save_csv(PER_BATCH_CSV, all_per_batch_records, per_batch_fields)

    # ========================================================
    # 10. 输出文字报告
    # ========================================================

    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("EndoPolyp-Seg Benchmark V3 Report\n")
        file.write("=" * 72 + "\n")
        file.write(f"GPU：{gpu_name}\n")
        file.write(f"PyTorch：{torch.__version__}\n")
        file.write(f"CUDA：{torch.version.cuda}\n")
        file.write(f"Batch Size：{BATCH_SIZE}\n")
        file.write(f"Input Size：{INPUT_SIZE[0]}x{INPUT_SIZE[1]}\n")
        file.write("Precision：FP32\n")
        file.write(f"cuDNN benchmark：{torch.backends.cudnn.benchmark}\n")
        file.write(f"Test samples：{len(test_loader.dataset)}\n")
        file.write(
            f"cuDNN shape warmup：{CUDNN_SHAPE_WARMUP_RUNS} runs per shape\n"
        )
        file.write(f"Model-only warmup：{MODEL_ONLY_WARMUP_RUNS}\n")
        file.write(f"Model-only repeats：{MODEL_ONLY_MEASURE_RUNS}\n")
        file.write(f"E2E warmup batches/model：{E2E_WARMUP_BATCHES}\n\n")

        file.write("Timing methodology\n")
        file.write("-" * 72 + "\n")
        file.write(
            "1. 所有模型沿用 V2 的结构、checkpoint、测试集和预处理配置。\n"
        )
        file.write(
            "2. cuDNN benchmark=True；正式测速前对完整 batch 和尾批 batch shape "
            "分别预热，避免首次遇到尾批 shape 时才进行算法搜索。\n"
        )
        file.write(
            "3. GPU model-only：H2D 和 normalization 在计时区间之外，"
            "只统计模型 forward。\n"
        )
        file.write(
            "4. E2E：每个 batch 统计 DataLoader next、图像 H2D、"
            "GPU normalization、GPU forward 和 CUDA synchronize。\n"
        )
        file.write(
            "5. 稳态统计只包含 batch size=4 且不是正式遍历首批的记录；"
            "首批和不完整尾批单独报告。\n"
        )
        file.write(
            "6. 所有测试图片（包括尾批）均参与逐图 Dice / IoU；"
            "mask 保留在 CPU，指标计算不计入推理耗时。\n"
        )
        file.write(
            "7. 完整遍历吞吐量包含首批与尾批；稳态吞吐量仅用于完整 batch 比较。\n"
        )
        file.write(
            "8. FLOPs 按 2×MACs 估算；不同工具的 FLOPs 统计口径可能不同。\n"
        )
        file.write(
            "9. 尾批预热是针对 V2 异常的诊断性修正；若仍有异常，需继续检查逐 Batch 记录。\n\n"
        )

        if skipped_models:
            file.write("Skipped models\n")
            for message in skipped_models:
                file.write(f"- {message}\n")
            file.write("\n")

        for result in summary_records:
            file.write("-" * 72 + "\n")
            file.write(f"Model：{result['model']}\n")
            file.write(f"Checkpoint：{result['checkpoint']}\n")
            file.write(f"Parameters：{result['parameters_m']:.3f} M\n")
            file.write(
                f"Checkpoint size：{result['checkpoint_size_mb']:.2f} MB\n"
            )

            if result["macs_per_image_g"] != "":
                file.write(
                    f"MACs：{result['macs_per_image_g']:.3f} G/image\n"
                )
                file.write(
                    f"FLOPs：{result['flops_per_image_g']:.3f} G/image\n"
                )
            else:
                file.write(f"Complexity：{result['complexity_note']}\n")

            file.write("\nGPU model-only (ms/batch, batch=4)\n")
            file.write(
                f"Mean={result['gpu_mean_batch_ms']:.3f}, "
                f"Median={result['gpu_median_batch_ms']:.3f}, "
                f"P95={result['gpu_p95_batch_ms']:.3f}, "
                f"Min={result['gpu_min_batch_ms']:.3f}, "
                f"Max={result['gpu_max_batch_ms']:.3f}, "
                f"Std={result['gpu_std_batch_ms']:.3f}\n"
            )
            file.write(
                f"Throughput={result['gpu_throughput_img_s']:.2f} images/s\n"
            )

            file.write("\nEnd-to-end timing\n")
            file.write(
                f"First batch={result['first_batch_e2e_ms']:.3f} ms\n"
            )
            file.write(
                f"Tail batch size={result['tail_batch_size']}, "
                f"latency={result['tail_batch_e2e_ms']:.3f} ms, "
                f"forward={result['tail_batch_forward_ms']:.3f} ms\n"
            )
            file.write(
                f"Steady full batches={result['steady_full_batch_count']}, "
                f"images={result['steady_full_image_count']}\n"
            )
            file.write(
                f"Steady Mean={result['e2e_steady_mean_batch_ms']:.3f}, "
                f"Median={result['e2e_steady_median_batch_ms']:.3f}, "
                f"P95={result['e2e_steady_p95_batch_ms']:.3f}, "
                f"Min={result['e2e_steady_min_batch_ms']:.3f}, "
                f"Max={result['e2e_steady_max_batch_ms']:.3f}, "
                f"Std={result['e2e_steady_std_batch_ms']:.3f} ms/batch\n"
            )
            file.write(
                f"Steady throughput="
                f"{result['e2e_steady_throughput_img_s']:.2f} images/s\n"
            )
            file.write(
                f"Full-pass throughput="
                f"{result['e2e_full_pass_throughput_img_s']:.2f} images/s\n"
            )

            file.write("\nSteady component means (ms/batch)\n")
            file.write(
                f"DataLoader + CPU preprocessing="
                f"{result['loader_wait_mean_ms']:.3f}\n"
            )
            file.write(f"H2D={result['h2d_mean_ms']:.3f}\n")
            file.write(
                f"GPU normalization={result['gpu_normalize_mean_ms']:.3f}\n"
            )
            file.write(
                f"GPU forward={result['gpu_forward_mean_ms']:.3f}\n"
            )

            file.write("\nTest metrics\n")
            file.write(f"Samples={result['test_samples']}\n")
            file.write(f"Dice={result['test_mean_dice']:.4f}\n")
            file.write(f"IoU={result['test_mean_iou']:.4f}\n")
            file.write(
                f"Peak allocated={result['peak_memory_allocated_gb']:.3f} GiB\n"
            )
            file.write(
                f"Peak reserved={result['peak_memory_reserved_gb']:.3f} GiB\n"
            )
            file.write("\n")

    print("\n" + "=" * 72)
    print("Benchmark V3 全部完成")
    print("=" * 72)
    print(f"汇总 CSV：{SUMMARY_CSV}")
    print(f"逐图 CSV：{PER_IMAGE_CSV}")
    print(f"逐 Batch CSV：{PER_BATCH_CSV}")
    print(f"文字报告：{REPORT_TXT}")


if __name__ == "__main__":
    main()
