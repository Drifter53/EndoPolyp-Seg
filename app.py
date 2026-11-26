
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ==========================================
# 1. 项目路径与设备配置
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = (256, 256)

# 推理速度测试配置
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

print("当前运行设备：", DEVICE)


# ==========================================
# 2. 导入三个模型
# ==========================================

from model import UNet as OriginalUNet
from model_resnet import ResNet34UNet
from model_attention import UNet as AttentionUNet


# ==========================================
# 3. 模型权重配置
# ==========================================

MODEL_CONFIGS = {
    "Original U-Net": {
        "checkpoint": PROJECT_ROOT / "outputs" / "best_unet.pth",
        "normalize_imagenet": False
    },

    "ResNet34-U-Net": {
        "checkpoint": PROJECT_ROOT / "outputs_resnet34" / "best_unet.pth",
        "normalize_imagenet": True
    },

    "Attention U-Net": {
        "checkpoint": PROJECT_ROOT / "outputs_attention" / "best_attention_unet.pth",
        "normalize_imagenet": True
    }
}


# ==========================================
# 4. 加载模型权重
# ==========================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        for key in [
            "model_state_dict",
            "state_dict",
            "model"
        ]:

            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def load_checkpoint(model, checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    # 兼容 DataParallel
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ==========================================
# 5. 创建三个模型
# ==========================================

def build_models():

    print("\n========== 加载三个模型 ==========")

    models = {}

    print("加载 Original U-Net...")
    original_model = OriginalUNet()

    original_model = load_checkpoint(
        original_model,
        MODEL_CONFIGS["Original U-Net"]["checkpoint"]
    )

    models["Original U-Net"] = original_model

    print("加载 ResNet34-U-Net...")

    try:
        resnet_model = ResNet34UNet(pretrained=False)
    except TypeError:
        resnet_model = ResNet34UNet()

    resnet_model = load_checkpoint(
        resnet_model,
        MODEL_CONFIGS["ResNet34-U-Net"]["checkpoint"]
    )

    models["ResNet34-U-Net"] = resnet_model

    print("加载 Attention U-Net...")
    attention_model = AttentionUNet()

    attention_model = load_checkpoint(
        attention_model,
        MODEL_CONFIGS["Attention U-Net"]["checkpoint"]
    )

    models["Attention U-Net"] = attention_model

    print("\n三个模型加载成功！")

    return models


MODELS = build_models()


# ==========================================
# 6. 图像预处理
# ==========================================

def preprocess_image(image, normalize_imagenet):

    image = image.convert("RGB")

    image_np = np.array(image)

    resized = cv2.resize(
        image_np,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    resized = resized.astype(np.float32) / 255.0

    # HWC -> CHW
    resized = np.transpose(
        resized,
        (2, 0, 1)
    ).copy()

    tensor = torch.from_numpy(resized).unsqueeze(0)

    # ImageNet 归一化
    if normalize_imagenet:

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        tensor = (tensor - mean) / std

    return tensor


# ==========================================
# 7. GPU 同步
# ==========================================

def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ==========================================
# 8. 单个模型推理
# ==========================================

def run_model(image, model_name):

    model = MODELS[model_name]

    normalize = MODEL_CONFIGS[model_name][
        "normalize_imagenet"
    ]

    original_np = np.array(
        image.convert("RGB")
    )

    height, width = original_np.shape[:2]

    input_tensor = preprocess_image(
        image,
        normalize
    ).to(DEVICE)

    try:

        # 一次只把一个模型放入 GPU
        model.to(DEVICE)
        model.eval()

        # 预热
        with torch.inference_mode():

            for _ in range(WARMUP_RUNS):
                _ = model(input_tensor)

        synchronize()

        # 正式计时
        times = []

        with torch.inference_mode():

            for _ in range(BENCHMARK_RUNS):

                synchronize()

                start = time.perf_counter()

                logits = model(input_tensor)

                # 兼容不同输出尺寸
                if logits.shape[-2:] != IMAGE_SIZE[::-1]:

                    logits = F.interpolate(
                        logits,
                        size=IMAGE_SIZE[::-1],
                        mode="bilinear",
                        align_corners=False
                    )

                synchronize()

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                times.append(elapsed)

            probability = torch.sigmoid(logits)

        probability_np = (
            probability[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # 二值化
        mask_small = (
            probability_np >= 0.5
        ).astype(np.uint8) * 255

        # 恢复原图尺寸
        mask = cv2.resize(
            mask_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        inference_time = float(np.median(times))

        return mask, inference_time

    finally:

        # 释放 GPU 显存
        model.cpu()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ==========================================
# 9. 生成红色叠加图
# ==========================================

def make_overlay(image, mask):

    original_np = np.array(
        image.convert("RGB")
    )

    overlay = original_np.copy()

    red = np.zeros_like(original_np)

    red[:, :, 0] = 255

    foreground = mask == 255

    overlay[foreground] = (
        0.65 * original_np[foreground]
        + 0.35 * red[foreground]
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# ==========================================
# 10. 处理人工标注
# ==========================================

def prepare_ground_truth(gt_image, height, width):

    if gt_image is None:
        return None

    gt = np.array(
        gt_image.convert("L")
    )

    if gt.shape != (height, width):

        gt = cv2.resize(
            gt,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    if gt.max() <= 1:
        gt = (gt > 0).astype(np.uint8)
    else:
        gt = (gt >= 128).astype(np.uint8)

    return gt


# ==========================================
# 11. Dice 和 IoU
# ==========================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction > 0

    gt = ground_truth > 0

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_sum = pred.sum()

    gt_sum = gt.sum()

    if pred_sum + gt_sum == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection
            / (pred_sum + gt_sum)
        )

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    return float(dice), float(iou)


# ==========================================
# 12. 单模型预测
# ==========================================

def predict_single(image, model_name, gt_image):

    if image is None:
        return None, None, None, None, "请先上传图片！"

    try:

        original = image.convert("RGB")

        height, width = np.array(original).shape[:2]

        mask, inference_time = run_model(
            original,
            model_name
        )

        overlay = make_overlay(
            original,
            mask
        )

        area_ratio = (
            np.count_nonzero(mask)
            / mask.size
        ) * 100

        gt = prepare_ground_truth(
            gt_image,
            height,
            width
        )

        if gt is not None:

            dice, iou = calculate_metrics(
                mask,
                gt * 255
            )

            gt_output = Image.fromarray(
                gt * 255
            )

            metric_text = f"""
| Dice | IoU |
|---|---|
| {dice:.4f} | {iou:.4f} |
"""

        else:

            gt_output = None

            metric_text = """
未上传人工标注，暂不计算 Dice 和 IoU。
"""

        result_text = f"""
### 单模型预测完成

| 项目 | 结果 |
|---|---|
| 使用模型 | {model_name} |
| 推理设备 | {DEVICE} |
| 原始图像尺寸 | {width} × {height} |
| 模型输入尺寸 | 256 × 256 |
| 平均推理耗时 | {inference_time:.2f} ms |
| 前景区域占比 | {area_ratio:.2f}% |
| 二值化阈值 | 0.5 |

### 分割性能

{metric_text}

**说明：** 推理耗时为预热后 5 次前向传播耗时的中位数，不包含图像预处理和设备传输。
"""

        return (
            original,
            gt_output,
            Image.fromarray(mask),
            overlay,
            result_text
        )

    except Exception as e:

        return (
            image,
            None,
            None,
            None,
            f"### 预测失败\n\n`{str(e)}`"
        )


# ==========================================
# 13. 三模型对照实验
# ==========================================

def compare_models(image, gt_image):

    if image is None:
        return (
            None, None,
            None, None, None,
            None, None, None,
            "请先上传图片！"
        )

    try:

        original = image.convert("RGB")

        height, width = np.array(original).shape[:2]

        gt = prepare_ground_truth(
            gt_image,
            height,
            width
        )

        gt_output = (
            Image.fromarray(gt * 255)
            if gt is not None
            else None
        )

        results = {}

        # 依次运行三个模型
        for model_name in MODEL_CONFIGS:

            print(f"\n正在测试：{model_name}")

            mask, inference_time = run_model(
                original,
                model_name
            )

            area_ratio = (
                np.count_nonzero(mask)
                / mask.size
            ) * 100

            overlay = make_overlay(
                original,
                mask
            )

            dice = None
            iou = None

            if gt is not None:

                dice, iou = calculate_metrics(
                    mask,
                    gt * 255
                )

            results[model_name] = {
                "mask": Image.fromarray(mask),
                "overlay": overlay,
                "time": inference_time,
                "area": area_ratio,
                "dice": dice,
                "iou": iou
            }

        # ==================================
        # 生成对照表
        # ==================================

        table = """
### 三模型对照实验结果

| 模型 | 推理耗时(ms) | 前景占比 | Dice | IoU |
|---|---:|---:|---:|---:|
"""

        for name, result in results.items():

            dice_text = (
                f"{result['dice']:.4f}"
                if result["dice"] is not None
                else "—"
            )

            iou_text = (
                f"{result['iou']:.4f}"
                if result["iou"] is not None
                else "—"
            )

            table += (
                f"| {name} "
                f"| {result['time']:.2f} "
                f"| {result['area']:.2f}% "
                f"| {dice_text} "
                f"| {iou_text} |\n"
            )

        table += f"""

### 实验配置

- 推理设备：{DEVICE}
- 原始图像尺寸：{width} × {height}
- 模型输入尺寸：256 × 256
- 二值化阈值：0.5
- 预热次数：{WARMUP_RUNS}
- 正式测试次数：{BENCHMARK_RUNS}

### 评价说明

推理时间采用预热后多次前向传播耗时的中位数。

"""

        if gt is None:

            table += """
当前没有提供人工标注，因此 Dice 和 IoU 不参与评价。
"""

        else:

            table += """
已提供人工标注，Dice 和 IoU 根据恢复到原始图像尺寸的预测 Mask 计算。
"""

        # ==================================
        # 返回所有结果
        # ==================================

        return (
            original,
            gt_output,

            results["Original U-Net"]["mask"],
            results["ResNet34-U-Net"]["mask"],
            results["Attention U-Net"]["mask"],

            results["Original U-Net"]["overlay"],
            results["ResNet34-U-Net"]["overlay"],
            results["Attention U-Net"]["overlay"],

            table
        )

    except Exception as e:

        return (
            image,
            None,
            None, None, None,
            None, None, None,
            f"### 对照实验失败\n\n`{str(e)}`"
        )


# ==========================================
# 14. 清空界面
# ==========================================

def clear_all():

    return (
        None,
        None,

        None, None, None, None,
        "",

        None, None,
        None, None, None,
        None, None, None,

        ""
    )


# ==========================================
# 15. Gradio 界面
# ==========================================

with gr.Blocks(
    title="EndoPolyp-Seg",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# EndoPolyp-Seg

### Deep Learning Based Polyp Segmentation System

基于深度学习的结肠镜息肉图像自动分割系统。

集成三种深度学习分割模型，支持单模型预测、
三模型对照、分割可视化和定量性能评价。
"""
    )

    gr.Markdown("---")

    with gr.Row():

        # ==================================
        # 左侧：输入
        # ==================================

        with gr.Column(scale=1):

            gr.Markdown("### 1. 输入图像")

            image_input = gr.Image(
                type="pil",
                label="上传结肠镜图像"
            )

            gt_input = gr.Image(
                type="pil",
                label="可选：上传人工标注 Mask"
            )

            model_choice = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="ResNet34-U-Net",
                label="单模型预测选择"
            )

            predict_button = gr.Button(
                "单模型预测",
                variant="primary"
            )

            compare_button = gr.Button(
                "三模型一键对照",
                variant="primary"
            )

            clear_button = gr.Button("清空")

        # ==================================
        # 右侧：结果
        # ==================================

        with gr.Column(scale=2):

            with gr.Tabs():

                # ----------------------------------
                # Tab 1：单模型预测
                # ----------------------------------

                with gr.Tab("单模型预测"):

                    gr.Markdown("### 单模型分割结果")

                    with gr.Row():

                        single_original = gr.Image(
                            label="Original Image",
                            type="pil"
                        )

                        single_gt = gr.Image(
                            label="Ground Truth",
                            type="pil"
                        )

                    with gr.Row():

                        single_mask = gr.Image(
                            label="Predicted Mask",
                            type="pil"
                        )

                        single_overlay = gr.Image(
                            label="Segmentation Overlay",
                            type="pil"
                        )

                    single_result = gr.Markdown(
                        "等待预测..."
                    )

                # ----------------------------------
                # Tab 2：三模型对照
                # ----------------------------------

                with gr.Tab("三模型对照"):

                    gr.Markdown(
                        "### 三模型预测结果对照"
                    )

                    with gr.Row():

                        compare_original = gr.Image(
                            label="Original Image",
                            type="pil"
                        )

                        compare_gt = gr.Image(
                            label="Ground Truth",
                            type="pil"
                        )

                    gr.Markdown("### 预测 Mask")

                    with gr.Row():

                        compare_mask_original = gr.Image(
                            label="Original U-Net",
                            type="pil"
                        )

                        compare_mask_resnet = gr.Image(
                            label="ResNet34-U-Net",
                            type="pil"
                        )

                        compare_mask_attention = gr.Image(
                            label="Attention U-Net",
                            type="pil"
                        )

                    gr.Markdown("### 预测叠加图")

                    with gr.Row():

                        compare_overlay_original = gr.Image(
                            label="Original U-Net Overlay",
                            type="pil"
                        )

                        compare_overlay_resnet = gr.Image(
                            label="ResNet34-U-Net Overlay",
                            type="pil"
                        )

                        compare_overlay_attention = gr.Image(
                            label="Attention U-Net Overlay",
                            type="pil"
                        )

                    compare_result = gr.Markdown(
                        "等待三模型对照实验..."
                    )

    # ==================================
    # 16. 事件绑定
    # ==================================

    predict_button.click(
        fn=predict_single,
        inputs=[
            image_input,
            model_choice,
            gt_input
        ],
        outputs=[
            single_original,
            single_gt,
            single_mask,
            single_overlay,
            single_result
        ]
    )

    compare_button.click(
        fn=compare_models,
        inputs=[
            image_input,
            gt_input
        ],
        outputs=[
            compare_original,
            compare_gt,

            compare_mask_original,
            compare_mask_resnet,
            compare_mask_attention,

            compare_overlay_original,
            compare_overlay_resnet,
            compare_overlay_attention,

            compare_result
        ]
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            image_input,
            gt_input,

            single_original,
            single_gt,
            single_mask,
            single_overlay,
            single_result,

            compare_original,
            compare_gt,

            compare_mask_original,
            compare_mask_resnet,
            compare_mask_attention,

            compare_overlay_original,
            compare_overlay_resnet,
            compare_overlay_attention,

            compare_result
        ]
    )

    gr.Markdown(
        """
---
**Models:** Original U-Net | ResNet34-U-Net | Attention U-Net

**Dataset:** Kvasir-SEG

**Framework:** PyTorch / OpenCV / Gradio

**Task:** Medical Image Segmentation
"""
    )


# ==========================================
# 17. 启动程序
# ==========================================

if __name__ == "__main__":

    print("\n========== EndoPolyp-Seg ==========")

    print("三个模型加载完成！")

    print("启动 Gradio Web 服务...")

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True
    )
