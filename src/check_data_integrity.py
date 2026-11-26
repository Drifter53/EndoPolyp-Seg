
from pathlib import Path
import csv
import cv2
import numpy as np

# ==========================================
# 1. 路径配置
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "Kvasir-SEG"

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_integrity"
PREVIEW_DIR = OUTPUT_DIR / "abnormal_samples"

# 前景比例检查阈值
MIN_FOREGROUND_RATIO = 0.001
MAX_FOREGROUND_RATIO = 0.80

# 最多保存多少张异常样本
MAX_PREVIEWS = 50


# ==========================================
# 2. 读取数据划分
# ==========================================

def read_split(name):

    split_file = SPLIT_DIR / f"{name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(split_file)

    samples = []

    with open(split_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"划分文件格式错误：{line}"
                )

            image_name, mask_name = parts

            samples.append((image_name, mask_name))

    return samples


# ==========================================
# 3. 检查单张样本
# ==========================================

def check_sample(
    split_name,
    index,
    image_name,
    mask_name,
    seen_images
):

    warnings = []

    image_path = IMAGE_DIR / image_name
    mask_path = MASK_DIR / mask_name

    # 检查是否重复划分
    if image_name in seen_images:

        warnings.append(
            f"重复划分，已出现在 {seen_images[image_name]}"
        )

    else:
        seen_images[image_name] = split_name

    # 检查文件是否存在
    if not image_path.exists():
        warnings.append("Image 文件不存在")

    if not mask_path.exists():
        warnings.append("Mask 文件不存在")

    if not image_path.exists() or not mask_path.exists():

        return {
            "split": split_name,
            "image_name": image_name,
            "mask_name": mask_name,
            "image_height": "",
            "image_width": "",
            "mask_unique_values": "",
            "foreground_ratio": "",
            "warnings": "; ".join(warnings)
        }, None

    # 读取原始文件
    image = cv2.imread(str(image_path))

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        warnings.append("Image 读取失败")

    if mask is None:
        warnings.append("Mask 读取失败")

    if image is None or mask is None:

        return {
            "split": split_name,
            "image_name": image_name,
            "mask_name": mask_name,
            "image_height": "",
            "image_width": "",
            "mask_unique_values": "",
            "foreground_ratio": "",
            "warnings": "; ".join(warnings)
        }, None

    # 检查尺寸
    if image.shape[:2] != mask.shape[:2]:

        warnings.append("Image 和 Mask 尺寸不一致")

    # Mask 二值化
    binary_mask = mask > 127

    # 计算前景比例
    foreground_ratio = float(binary_mask.mean())

    # 检查空白 Mask
    if foreground_ratio == 0:

        warnings.append("Mask 完全为空")

    elif foreground_ratio < MIN_FOREGROUND_RATIO:

        warnings.append("前景区域极小，需要人工复查")

    # 检查异常大的前景
    if foreground_ratio > MAX_FOREGROUND_RATIO:

        warnings.append("前景区域异常大，需要人工复查")

    # 记录 Mask 像素信息
    unique_values = len(np.unique(mask))

    result = {

        "split": split_name,

        "image_name": image_name,

        "mask_name": mask_name,

        "image_height": image.shape[0],

        "image_width": image.shape[1],

        "mask_unique_values": unique_values,

        "foreground_ratio": round(
            foreground_ratio, 6
        ),

        "warnings": "; ".join(warnings)
    }

    # ======================================
    # 4. 保存异常样本可视化
    # ======================================

    preview = None

    if warnings:

        image_show = cv2.resize(
            image,
            (512, 512)
        )

        mask_show = cv2.resize(
            binary_mask.astype(np.uint8) * 255,
            (512, 512),
            interpolation=cv2.INTER_NEAREST
        )

        mask_bgr = cv2.cvtColor(
            mask_show,
            cv2.COLOR_GRAY2BGR
        )

        # 红色 Mask 叠加
        red = np.zeros_like(image_show)

        red[:, :, 2] = 255

        overlay = image_show.copy()

        foreground = mask_show > 0

        blended = cv2.addWeighted(
            image_show, 0.7,
            red, 0.3,
            0
        )

        overlay[foreground] = blended[foreground]

        preview = cv2.hconcat([
            image_show,
            mask_bgr,
            overlay
        ])

    return result, preview


# ==========================================
# 5. 主程序
# ==========================================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    seen_images = {}

    abnormal_count = 0

    preview_count = 0

    total_count = 0

    print("\n========== 数据完整性检查 ==========\n")

    for split_name in ["train", "val", "test"]:

        samples = read_split(split_name)

        print(
            f"\n正在检查 {split_name}，"
            f"样本数量：{len(samples)}"
        )

        for index, (image_name, mask_name) in enumerate(samples):

            result, preview = check_sample(
                split_name,
                index,
                image_name,
                mask_name,
                seen_images
            )

            all_results.append(result)

            total_count += 1

            if result["warnings"]:

                abnormal_count += 1

                print(
                    f"[异常] {split_name}: {image_name}"
                )

                print(
                    "原因：", result["warnings"]
                )

                if preview is not None and preview_count < MAX_PREVIEWS:

                    save_path = (
                        PREVIEW_DIR /
                        f"{split_name}_{index:04d}.jpg"
                    )

                    cv2.imwrite(
                        str(save_path),
                        preview
                    )

                    preview_count += 1

    # ======================================
    # 6. 保存 CSV 检查报告
    # ======================================

    report_path = (
        OUTPUT_DIR / "data_integrity_report.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_name",
                "mask_name",
                "image_height",
                "image_width",
                "mask_unique_values",
                "foreground_ratio",
                "warnings"
            ]
        )

        writer.writeheader()

        writer.writerows(all_results)

    # ======================================
    # 7. 输出检查结果
    # ======================================

    print("\n========== 检查结果 ==========")

    print("总样本数量：", total_count)

    print("异常样本数量：", abnormal_count)

    print("正常样本数量：", total_count - abnormal_count)

    print("异常可视化数量：", preview_count)

    print("检查报告：", report_path)

    print("异常图片：", PREVIEW_DIR)

    print("\n数据完整性检查结束！")
