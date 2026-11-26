"""
EndoPolyp-Seg Benchmark V2
==========================

V2 在 V1 基础上修正计时口径：
1. GPU model-only：真实测试集首批图片，先预热，再只测 GPU 前向。
2. End-to-end：每个模型单独预热 DataLoader 和模型，再重新遍历完整 test split。
3. 正式遍历的第一个 batch 单独记录为 post-warmup first batch，不纳入稳态统计。
4. 单独记录 DataLoader/CPU preprocessing wait、H2D、GPU normalization、GPU forward。
5. 测试集 Dice/IoU 仍在完整 test split 上逐图计算一次。
6. 保留参数量、Checkpoint 大小、MACs/FLOPs、峰值显存。

运行：
    python src/benchmark_v2.py

依赖：
    pip install thop

说明：
- 使用项目已有 dataset.py、model_resnet.py、model_attention.py。
- Original U-Net 按 V1/compare_models.py 的约定：src/model.py 中的 UNet，
  权重 outputs/best_unet.pth，不做 ImageNet normalization。
- ResNet34-U-Net、Attention U-Net 使用 ImageNet normalization。
- 原始 U-Net 训练脚本缺失，归一化约定尚未从训练脚本独立核实。
- V2 不修改模型权重和训练代码。
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

# 每个模型正式评估前，先读取并推理若干个测试 batch 作为预热。
E2E_WARMUP_BATCHES = 3

THRESHOLD = 0.5
SMOOTH = 1e-7

OUTPUT_DIR = PROJECT_ROOT / "outputs_benchmark_v2"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_v2_batch4.csv"
PER_IMAGE_CSV = OUTPUT_DIR / "benchmark_v2_per_image.csv"
PER_BATCH_CSV = OUTPUT_DIR / "benchmark_v2_per_batch.csv"
REPORT_TXT = OUTPUT_DIR / "benchmark_v2_report.txt"


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

    # 兼容部分 DataParallel 保存的 module. 前缀。
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
    """
    logits 从 GPU 拷贝到 CPU 后计算指标。
    这样 Dice/IoU 的计算和 mask 的 H2D 不会混入推理计时。
    """
    predictions = (torch.sigmoid(logits.detach().float().cpu()) >= THRESHOLD).float()
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
# 3. 模型配置
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
# 4. GPU-only 前向测速
# ============================================================

def benchmark_model_only(
    model,
    first_images_cpu,
    normalize_imagenet,
    device
):
    """
    只计 GPU 模型 forward：
    - 测试图片来自真实 test split；
    - H2D 和 normalization 在计时区间之外；
    - 先进行独立预热，再正式计时。
    """
    images = first_images_cpu.to(device, non_blocking=False)

    if normalize_imagenet:
        images = normalize_image(images)

    model.eval()

    with torch.inference_mode():
        for _ in range(MODEL_ONLY_WARMUP_RUNS):
            _ = model(images)

        torch.cuda.synchronize()

        times_ms = []
        for _ in range(MODEL_ONLY_MEASURE_RUNS):
            start_event, end_event = cuda_event_pair()
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
# 5. 端到端预热
# ============================================================

def warmup_end_to_end(model, test_loader, normalize_imagenet, device):
    """
    每个模型独立预热：
    - 重新创建 DataLoader iterator；
    - 读取最多 E2E_WARMUP_BATCHES 个真实测试 batch；
    - 执行 H2D、归一化和 forward；
    - 不计算 Dice/IoU，不记录为正式测试耗时。
    """
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
# 6. 端到端正式测试
# ============================================================

def benchmark_end_to_end(
    model,
    test_loader,
    normalize_imagenet,
    device
):
    """
    正式遍历完整 test split 一次。

    每个 batch 的计时：
      wall time = DataLoader next() + H2D(image) + GPU normalize
                  + GPU forward + CUDA synchronize

    额外拆分：
      loader_wait_ms：next(iterator)，包括 Dataset 的读取/解码/CPU预处理
      h2d_ms：输入图像 CPU -> GPU
      normalize_ms：GPU 归一化
      forward_ms：GPU 模型前向

    正式遍历的第一个 batch 单独报告为 post-warmup first batch，
    不纳入稳态 mean/median/p95，避免 iterator 重建/首批读取扰动稳态统计。
    第一个 batch 仍参与完整测试集 Dice/IoU。
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

            # 指标计算在计时结束后执行；mask 保留在 CPU，不做 mask H2D。
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
            is_startup_batch = (batch_index == 1)

            batch_record = {
                "batch_index": batch_index,
                "batch_size": current_batch_size,
                "batch_role": (
                    "post_warmup_first_batch"
                    if is_startup_batch else "steady_state_candidate"
                ),
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

    # 第一个正式 batch 单独记录；其余 batch 用于稳态统计。
    startup_record = all_batch_records[0]
    steady_records = all_batch_records[1:]

    def get_component_stats(records, key):
        return summarize_times([record[key] for record in records])

    steady_e2e_times = [
        record["e2e_latency_ms"] for record in steady_records
    ]
    full_pass_times = [
        record["e2e_latency_ms"] for record in all_batch_records
    ]

    steady_images = sum(
        record["batch_size"] for record in steady_records
    )
    steady_total_ms = sum(steady_e2e_times)
    full_total_ms = sum(full_pass_times)

    result = {
        "iterator_init_ms": iterator_init_ms,
        "startup_batch_e2e_ms": startup_record["e2e_latency_ms"],
        "startup_batch_loader_wait_ms": startup_record["loader_wait_ms"],
        "startup_batch_h2d_ms": startup_record["h2d_ms"],
        "startup_batch_normalize_ms": startup_record["gpu_normalize_ms"],
        "startup_batch_forward_ms": startup_record["gpu_forward_ms"],

        "steady_batch_count": len(steady_records),
        "steady_image_count": steady_images,
        "test_samples": total_samples,
        "test_batches": len(all_batch_records),

        "steady_e2e": summarize_times(steady_e2e_times),
        "steady_loader_wait": get_component_stats(
            steady_records, "loader_wait_ms"
        ),
        "steady_h2d": get_component_stats(steady_records, "h2d_ms"),
        "steady_normalize": get_component_stats(
            steady_records, "gpu_normalize_ms"
        ),
        "steady_forward": get_component_stats(
            steady_records, "gpu_forward_ms"
        ),

        "steady_throughput_img_s": (
            steady_images * 1000.0 / steady_total_ms
            if steady_total_ms > 0 else float("nan")
        ),
        "steady_amortized_ms_per_image": (
            steady_total_ms / steady_images
            if steady_images > 0 else float("nan")
        ),
        "full_pass_throughput_img_s": (
            total_samples * 1000.0 / full_total_ms
            if full_total_ms > 0 else float("nan")
        ),

        "mean_dice": float(np.mean(all_dice)),
        "mean_iou": float(np.mean(all_iou)),
    }

    return result, per_image_records, all_batch_records


# ============================================================
# 7. 主程序
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。本脚本要求使用 CUDA，不会自动切换到 CPU。\n"
            "请使用项目的 .venv-gpu311 Python 解释器运行。"
        )

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    device = DEVICE
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 72)
    print("EndoPolyp-Seg Benchmark V2")
    print("=" * 72)
    print(f"GPU：{gpu_name}")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"Batch Size：{BATCH_SIZE}")
    print(f"输入尺寸：{INPUT_SIZE[0]} x {INPUT_SIZE[1]}")
    print("精度：FP32")
    print(f"GPU-only 预热：{MODEL_ONLY_WARMUP_RUNS} 次")
    print(f"GPU-only 正式计时：{MODEL_ONLY_MEASURE_RUNS} 次")
    print(f"端到端预热：每模型最多 {E2E_WARMUP_BATCHES} 个 batch")
    print("端到端正式测试：完整 test split 一遍")
    print(f"输出目录：{OUTPUT_DIR}")

    # 复用项目已有 Dataset / DataLoader，所有模型使用相同 test split。
    _train_loader, _val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize_imagenet=False
    )

    if len(test_loader.dataset) == 0:
        raise RuntimeError("测试集为空，请检查 Kvasir-SEG test split。")

    if len(test_loader) < 2:
        raise RuntimeError(
            "测试集 batch 数少于 2，无法把首批与稳态 batch 分开统计。"
        )

    if not isinstance(test_loader.sampler, SequentialSampler):
        raise RuntimeError(
            "当前 test_loader 不是顺序采样。为保证逐图指标与文件名对应，"
            "请确认测试集 DataLoader 使用 shuffle=False。"
        )

    if getattr(test_loader.dataset, "augment", False):
        raise RuntimeError("检测到测试集启用了数据增强，请关闭 test augmentation。")

    if getattr(test_loader, "drop_last", False):
        raise RuntimeError("test_loader.drop_last=True，会丢弃测试集尾部图片。")

    first_images, _first_masks = next(iter(test_loader))
    if tuple(first_images.shape[-2:]) != INPUT_SIZE:
        raise RuntimeError(
            f"Dataset 实际输出尺寸为 {tuple(first_images.shape[-2:])}，"
            f"预期为 {INPUT_SIZE}"
        )

    print(f"\n测试集图片数：{len(test_loader.dataset)}")
    print(f"测试集 batch 数：{len(test_loader)}")
    print(f"首批输入形状：{tuple(first_images.shape)}")

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

            # 复杂度分析的临时显存不计入推理峰值。
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            print(f"参数量：{total_params / 1e6:.2f} M")
            print(f"Checkpoint：{ckpt_mb:.2f} MB")
            if macs is not None:
                print(f"MACs：{macs / 1e9:.2f} G / image")
                print(f"FLOPs：{flops / 1e9:.2f} G / image")
            else:
                print(f"复杂度统计：{complexity_note}")

            # A. 纯 GPU forward 测速
            print("\nA. GPU model-only 测速（真实测试集首批图片）")
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

            # B. 每个模型单独进行端到端预热
            print("\nB. 端到端预热")
            warmed_batches = warmup_end_to_end(
                model,
                test_loader,
                spec["normalize_imagenet"],
                device
            )
            print(f"  已完成 {warmed_batches} 个 warm-up batch")

            # C. 完整 test split 一次，第一批单独列出
            print("\nC. 端到端正式测试")
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

            print(
                f"  DataLoader iterator 初始化："
                f"{e2e['iterator_init_ms']:.3f} ms（单独记录）"
            )
            print(
                f"  预热后首批延迟："
                f"{e2e['startup_batch_e2e_ms']:.3f} ms（不纳入稳态统计）"
            )

            steady = e2e["steady_e2e"]
            print(
                f"  稳态 E2E Mean / Median / P95："
                f"{steady['mean']:.3f} / "
                f"{steady['median']:.3f} / "
                f"{steady['p95']:.3f} ms / batch"
            )
            print(
                f"  稳态 E2E Min / Max / Std："
                f"{steady['min']:.3f} / "
                f"{steady['max']:.3f} / "
                f"{steady['std']:.3f} ms"
            )
            print(
                f"  稳态吞吐量："
                f"{e2e['steady_throughput_img_s']:.2f} images/s"
            )
            print(
                f"  稳态平均单图耗时："
                f"{e2e['steady_amortized_ms_per_image']:.3f} ms/image"
            )

            print("\n  稳态分项耗时（每 batch 的平均值）：")
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

            print(
                f"\n  完整 test split Dice：{e2e['mean_dice']:.4f}"
            )
            print(f"  完整 test split IoU：{e2e['mean_iou']:.4f}")
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
                "e2e_startup_batch_ms": e2e["startup_batch_e2e_ms"],
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
                "steady_batch_count": e2e["steady_batch_count"],
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

    # --------------------------------------------------------
    # 输出 CSV
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 输出文字报告
    # --------------------------------------------------------

    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("EndoPolyp-Seg Benchmark V2 Report\n")
        file.write("=" * 72 + "\n")
        file.write(f"GPU：{gpu_name}\n")
        file.write(f"PyTorch：{torch.__version__}\n")
        file.write(f"CUDA：{torch.version.cuda}\n")
        file.write(f"Batch Size：{BATCH_SIZE}\n")
        file.write(f"Input Size：{INPUT_SIZE[0]}x{INPUT_SIZE[1]}\n")
        file.write("Precision：FP32\n")
        file.write(f"Test samples：{len(test_loader.dataset)}\n")
        file.write(f"Model-only warmup：{MODEL_ONLY_WARMUP_RUNS}\n")
        file.write(f"Model-only repeats：{MODEL_ONLY_MEASURE_RUNS}\n")
        file.write(f"E2E warmup batches/model：{E2E_WARMUP_BATCHES}\n\n")

        file.write("Timing methodology\n")
        file.write("-" * 72 + "\n")
        file.write(
            "1. GPU model-only：先将真实测试 batch 传入 GPU 并完成归一化，"
            "再测纯模型 forward；H2D 和 normalization 不计入该项。\n"
        )
        file.write(
            "2. E2E：每个模型先独立预热，再重新遍历完整 test split；"
            "计时包括 DataLoader next、图像 H2D、GPU normalization、"
            "GPU forward 和 CUDA synchronize。\n"
        )
        file.write(
            "3. 正式遍历的第一个 batch 单独记录，不纳入稳态 mean/median/P95；"
            "但仍参与完整 test split 的 Dice/IoU。\n"
        )
        file.write(
            "4. loader_wait 包括 Dataset 的磁盘读取、图像解码和 CPU 预处理；"
            "这些操作在当前 Dataset 的 __getitem__ 内执行，无法再无侵入地拆分。\n"
        )
        file.write(
            "5. masks 留在 CPU；mask H2D 和 Dice/IoU 计算不计入推理耗时。\n"
        )
        file.write(
            "6. FLOPs 按 2×MACs 估算；不同工具的 FLOPs 统计口径可能不同。\n\n"
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
            file.write(
                f"Parameters：{result['parameters_m']:.3f} M\n"
            )
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
                file.write(
                    f"Complexity：{result['complexity_note']}\n"
                )

            file.write("\nGPU model-only (ms/batch)\n")
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

            file.write("\nEnd-to-end steady state (ms/batch)\n")
            file.write(
                f"Iterator init (separate)={result['iterator_init_ms']:.3f}\n"
            )
            file.write(
                f"First batch after warmup (separate)="
                f"{result['e2e_startup_batch_ms']:.3f}\n"
            )
            file.write(
                f"Mean={result['e2e_steady_mean_batch_ms']:.3f}, "
                f"Median={result['e2e_steady_median_batch_ms']:.3f}, "
                f"P95={result['e2e_steady_p95_batch_ms']:.3f}, "
                f"Min={result['e2e_steady_min_batch_ms']:.3f}, "
                f"Max={result['e2e_steady_max_batch_ms']:.3f}, "
                f"Std={result['e2e_steady_std_batch_ms']:.3f}\n"
            )
            file.write(
                f"Steady throughput="
                f"{result['e2e_steady_throughput_img_s']:.2f} images/s\n"
            )
            file.write(
                f"Steady amortized latency="
                f"{result['e2e_steady_mean_per_image_ms']:.3f} ms/image\n"
            )
            file.write(
                f"Full pass throughput (includes first formal batch)="
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
    print("Benchmark V2 全部完成")
    print("=" * 72)
    print(f"汇总 CSV：{SUMMARY_CSV}")
    print(f"逐图 CSV：{PER_IMAGE_CSV}")
    print(f"逐 Batch CSV：{PER_BATCH_CSV}")
    print(f"文字报告：{REPORT_TXT}")


if __name__ == "__main__":
    main()
