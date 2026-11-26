from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# ==================== 1. 路径配置 ====================

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "outputs/test_results_original_unet/all_image_metrics.csv"

V4_CSV = ROOT / "outputs_benchmark_v4_batch1/benchmark_v4_batch1_per_image.csv"

DATA_DIR = ROOT / "data/Kvasir-SEG"
MASK_DIR = DATA_DIR / "masks"
BBOX_JSON = DATA_DIR / "kavsir_bboxes.json"

OUT_DIR = ROOT / "outputs_difficulty_analysis"

MODELS = [
    "Original U-Net",
    "ResNet34-U-Net",
    "Attention U-Net"
]

SIZE_NAMES = ["Small", "Medium", "Large"]

RESIZE = (256, 256)


# ==================== 2. 查找 Mask ====================

def one_mask_for_stem(stem):

    files = [
        p for p in MASK_DIR.iterdir()
        if p.is_file() and p.stem == stem
    ]

    if len(files) != 1:
        raise RuntimeError(
            f"{stem}: expected exactly one mask, found {len(files)}"
        )

    return files[0]


# ==================== 3. 从真实 Mask 提取特征 ====================

def mask_features(path):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")

    original_h, original_w = mask.shape

    # 与 dataset.py 保持一致
    resized = cv2.resize(
        mask,
        RESIZE,
        interpolation=cv2.INTER_NEAREST
    )

    foreground = resized > 127

    pixels = int(foreground.sum())

    if pixels == 0:
        raise RuntimeError(
            f"Empty mask: {path.name}"
        )

    # 获取所有前景像素坐标
    ys, xs = np.where(foreground)

    # Mask 外接矩形
    x1 = int(xs.min())
    x2 = int(xs.max())

    y1 = int(ys.min())
    y2 = int(ys.max())

    bbox_width = x2 - x1 + 1
    bbox_height = y2 - y1 + 1

    # Mask 质心
    cx = float(xs.mean() + 0.5)
    cy = float(ys.mean() + 0.5)

    # 归一化中心坐标
    cx_norm = cx / RESIZE[1]
    cy_norm = cy / RESIZE[0]

    # Mask 连通区域
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8),
            connectivity=8
        )
    )

    component_count = num_labels - 1

    return {

        "image_width": original_w,
        "image_height": original_h,

        # 前景面积
        "mask_foreground_pixels_256": pixels,

        "mask_foreground_area_ratio_256": (
            pixels / (RESIZE[0] * RESIZE[1])
        ),

        # Mask 外接矩形
        "mask_bbox_xmin_256": x1,
        "mask_bbox_ymin_256": y1,
        "mask_bbox_xmax_256": x2,
        "mask_bbox_ymax_256": y2,

        "mask_bbox_width_256": bbox_width,
        "mask_bbox_height_256": bbox_height,

        "mask_bbox_area_ratio_256": (
            bbox_width * bbox_height /
            (RESIZE[0] * RESIZE[1])
        ),

        "mask_bbox_width_ratio": (
            bbox_width / RESIZE[1]
        ),

        "mask_bbox_height_ratio": (
            bbox_height / RESIZE[0]
        ),

        # Mask 质心
        "mask_centroid_x_256": cx,
        "mask_centroid_y_256": cy,

        "mask_centroid_x_norm": cx_norm,
        "mask_centroid_y_norm": cy_norm,

        # 连通区域数量
        "mask_connected_components": component_count
    }


# ==================== 4. JSON 边界框辅助分析 ====================

def bbox_features(record, h, w, image_name):

    if (
        int(record["height"]) != h
        or int(record["width"]) != w
    ):
        raise RuntimeError(
            f"{image_name}: bbox JSON dimensions mismatch."
        )

    boxes = record.get("bbox")

    if not isinstance(boxes, list):
        raise RuntimeError(
            f"{image_name}: invalid bbox list."
        )

    boxes = [
        b for b in boxes
        if isinstance(b, dict)
        and str(b.get("label", "")).lower() == "polyp"
    ]

    valid_boxes = []
    invalid_count = 0
    clipped = False

    for b in boxes:

        try:

            raw = [
                float(b["xmin"]),
                float(b["ymin"]),
                float(b["xmax"]),
                float(b["ymax"])
            ]

        except (KeyError, TypeError, ValueError):

            invalid_count += 1
            continue

        if not np.isfinite(raw).all():

            invalid_count += 1
            continue

        x1, y1, x2, y2 = raw

        x1c, x2c = np.clip(
            [x1, x2], 0, w
        )

        y1c, y2c = np.clip(
            [y1, y2], 0, h
        )

        if any(
            abs(a - c) > 1e-9
            for a, c in zip(
                raw,
                [x1c, y1c, x2c, y2c]
            )
        ):

            clipped = True

        if x2c <= x1c or y2c <= y1c:

            invalid_count += 1
            continue

        valid_boxes.append(
            (x1c, y1c, x2c, y2c)
        )

    # 多个有效框取并集，仅作为辅助信息
    if valid_boxes:

        x1 = min(b[0] for b in valid_boxes)
        y1 = min(b[1] for b in valid_boxes)

        x2 = max(b[2] for b in valid_boxes)
        y2 = max(b[3] for b in valid_boxes)

        bw = x2 - x1
        bh = y2 - y1

        bbox_area = bw * bh

        bbox_center_x = ((x1 + x2) / 2) / w
        bbox_center_y = ((y1 + y2) / 2) / h

    else:

        x1 = y1 = x2 = y2 = np.nan
        bw = bh = bbox_area = np.nan

        bbox_center_x = np.nan
        bbox_center_y = np.nan

    return {

        "json_bbox_count": len(boxes),

        "json_bbox_valid_count": len(valid_boxes),

        "json_bbox_invalid_count": invalid_count,

        "json_bbox_xmin": x1,
        "json_bbox_ymin": y1,
        "json_bbox_xmax": x2,
        "json_bbox_ymax": y2,

        "json_bbox_width": bw,
        "json_bbox_height": bh,

        "json_bbox_area_pixels": bbox_area,

        "json_bbox_area_ratio": (
            bbox_area / (w * h)
            if np.isfinite(bbox_area)
            else np.nan
        ),

        "json_bbox_center_x_norm": bbox_center_x,
        "json_bbox_center_y_norm": bbox_center_y,

        "json_bbox_was_clipped": clipped
    }


# ==================== 5. 主程序 ====================

def main():

    print("=" * 60)
    print("EndoPolyp-Seg Difficulty Analysis")
    print("=" * 60)

    for p in [TEST_CSV, V4_CSV, BBOX_JSON, MASK_DIR]:

        if not p.exists():
            raise FileNotFoundError(p)

    # 读取固定的150张测试图像
    test = pd.read_csv(
        TEST_CSV,
        encoding="utf-8-sig"
    )

    if (
        len(test) != 150
        or test.image_name.nunique() != 150
    ):
        raise RuntimeError(
            "Expected exactly 150 unique images."
        )

    names = test.image_name.astype(str).tolist()

    stems = [
        Path(n).stem for n in names
    ]

    # 读取 JSON
    with BBOX_JSON.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        bbox_data = json.load(f)

    if not isinstance(bbox_data, dict):
        raise RuntimeError("Invalid bbox JSON.")

    # 提取150张图像特征
    rows = []

    for index, (name, stem) in enumerate(
        zip(names, stems), 1
    ):

        if stem not in bbox_data:
            raise RuntimeError(
                f"Missing bbox JSON: {name}"
            )

        mask_path = one_mask_for_stem(stem)

        # 真实 Mask 特征
        mask_info = mask_features(mask_path)

        # JSON 边界框辅助信息
        bbox_info = bbox_features(
            bbox_data[stem],
            mask_info["image_height"],
            mask_info["image_width"],
            name
        )

        rows.append({

            "image_name": name,

            "mask_name": mask_path.name,

            **mask_info,

            **bbox_info

        })

        if index % 30 == 0:
            print(
                f"已处理图像：{index}/150"
            )

    features = pd.DataFrame(rows)

    # ==================== 6. 检查三个模型的数据 ====================

    v4 = pd.read_csv(
        V4_CSV,
        encoding="utf-8-sig"
    )

    required = {
        "model",
        "image_name",
        "dice",
        "iou"
    }

    if not required.issubset(v4.columns):

        raise RuntimeError(
            "V4 CSV is missing required columns."
        )

    if set(v4.model.unique()) != set(MODELS):

        raise RuntimeError(
            f"Unexpected models: {v4.model.unique().tolist()}"
        )

    if (
        len(v4) != 450
        or v4.duplicated(
            ["model", "image_name"]
        ).any()
    ):

        raise RuntimeError(
            "Expected 450 unique model-image rows."
        )

    expected = {
        (m, n)
        for m in MODELS
        for n in names
    }

    actual = set(
        zip(
            v4.model.astype(str),
            v4.image_name.astype(str)
        )
    )

    if actual != expected:

        raise RuntimeError(
            "Model-image mismatch."
        )

    v4[["dice", "iou"]] = v4[
        ["dice", "iou"]
    ].apply(
        pd.to_numeric,
        errors="raise"
    )

    metrics = v4[["dice", "iou"]].to_numpy()

    if not np.isfinite(metrics).all():

        raise RuntimeError(
            "Non-finite Dice/IoU."
        )

    if ((metrics < 0) | (metrics > 1)).any():

        raise RuntimeError(
            "Dice/IoU must be in [0,1]."
        )

    # ==================== 7. 按真实 Mask 面积分组 ====================

    area = features[
        "mask_foreground_area_ratio_256"
    ].to_numpy()

    q1, q2 = np.quantile(
        area,
        [1/3, 2/3],
        method="linear"
    )

    if not q1 < q2:

        raise RuntimeError(
            "Tertile cut points are tied."
        )

    features["size_group"] = np.select(

        [
            features[
                "mask_foreground_area_ratio_256"
            ] <= q1,

            features[
                "mask_foreground_area_ratio_256"
            ] <= q2
        ],

        ["Small", "Medium"],

        default="Large"
    )

    size_counts = features[
        "size_group"
    ].value_counts().reindex(
        SIZE_NAMES,
        fill_value=0
    )

    if (size_counts == 0).any():

        raise RuntimeError(
            f"Empty size group: {size_counts.to_dict()}"
        )

    # ==================== 8. 按真实 Mask 质心划分位置 ====================

    cx = features["mask_centroid_x_norm"]
    cy = features["mask_centroid_y_norm"]

    center = (

        cx.ge(0.25)
        & cx.lt(0.75)

        & cy.ge(0.25)
        & cy.lt(0.75)

    )

    features["position_group"] = np.where(
        center,
        "Center",
        "Peripheral"
    )

    position_counts = features[
        "position_group"
    ].value_counts().reindex(
        ["Center", "Peripheral"],
        fill_value=0
    )

    if (position_counts == 0).any():

        raise RuntimeError(
            f"Empty position group: {position_counts.to_dict()}"
        )

    # ==================== 9. 合并模型指标 ====================

    combined = v4.merge(

        features[
            [
                "image_name",
                "size_group",
                "position_group"
            ]
        ],

        on="image_name",

        how="left",

        validate="many_to_one"
    )

    if (
        len(combined) != 450
        or combined[
            ["size_group", "position_group"]
        ].isna().any().any()
    ):

        raise RuntimeError(
            "Failed to join model metrics."
        )

    # ==================== 10. 不同尺寸的模型表现 ====================

    size_summary = (

        combined.groupby(
            ["size_group", "model"],
            observed=False
        )

        .agg(

            image_count=(
                "image_name",
                "nunique"
            ),

            mean_dice=(
                "dice",
                "mean"
            ),

            mean_iou=(
                "iou",
                "mean"
            )

        )

        .reset_index()
    )

    # ==================== 11. 不同位置的模型表现 ====================

    position_summary = (

        combined.groupby(
            ["position_group", "model"],
            observed=False
        )

        .agg(

            image_count=(
                "image_name",
                "nunique"
            ),

            mean_dice=(
                "dice",
                "mean"
            ),

            mean_iou=(
                "iou",
                "mean"
            )

        )

        .reset_index()
    )

    # ==================== 12. 保存结果 ====================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(

        OUT_DIR / "per_image_features.csv",

        index=False,

        encoding="utf-8-sig"
    )

    combined.to_csv(

        OUT_DIR / "per_image_model_metrics.csv",

        index=False,

        encoding="utf-8-sig"
    )

    size_summary.to_csv(

        OUT_DIR / "size_group_metrics.csv",

        index=False,

        encoding="utf-8-sig"
    )

    position_summary.to_csv(

        OUT_DIR / "position_group_metrics.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ==================== 13. 统计 JSON 异常 ====================

    clipped = int(
        features[
            "json_bbox_was_clipped"
        ].sum()
    )

    multi_bbox = int(
        (
            features["json_bbox_count"] > 1
        ).sum()
    )

    no_valid_bbox = int(
        (
            features["json_bbox_valid_count"] == 0
        ).sum()
    )

    # ==================== 14. 生成报告 ====================

    report = [

        "EndoPolyp-Seg Difficulty Analysis Report",

        "=" * 64,

        "Fixed existing test split: 150 images.",

        "Mask: grayscale -> 256x256 INTER_NEAREST -> mask > 127.",

        f"Images: {len(features)}",

        f"V4 model-image rows: {len(v4)}",

        f"Models: {', '.join(MODELS)}",

        "",

        "PRIMARY FEATURES",

        "Size: Ground Truth Mask foreground area ratio.",

        "Position: Ground Truth Mask centroid.",

        "Geometry: Ground Truth Mask bounding rectangle.",

        "JSON bounding boxes are auxiliary only.",

        "",

        "JSON BBOX DIAGNOSTICS",

        f"Images with multiple boxes: {multi_bbox}",

        f"Images with no valid bbox: {no_valid_bbox}",

        f"Images with clipped bbox: {clipped}",

        "",

        "SIZE GROUPS",

        "Empirical tertiles of Mask foreground area ratio.",

        f"Q1={q1:.10f}; Q2={q2:.10f}",

        "Small <= Q1; Medium > Q1 and <= Q2; Large > Q2",

        f"Counts: {size_counts.to_dict()}",

        size_summary.to_string(index=False),

        "",

        "POSITION GROUPS",

        "Based on normalized Mask centroid.",

        "Center: 0.25 <= cx < 0.75 and 0.25 <= cy < 0.75",

        "Peripheral: otherwise",

        f"Counts: {position_counts.to_dict()}",

        position_summary.to_string(index=False),

        "",

        "Visual background types require human review."

    ]

    (
        OUT_DIR / "difficulty_analysis_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # ==================== 15. 输出检查结果 ====================

    print("\n" + "=" * 60)

    print("Analysis completed!")

    print("=" * 60)

    print("测试图像数量:", len(features))

    print("模型指标记录:", len(v4))

    print("\n息肉尺寸分组:")

    print(size_counts.to_string())

    print("\n息肉位置分组:")

    print(position_counts.to_string())

    print("\nJSON 边界框检查:")

    print("多边界框图像数量:", multi_bbox)

    print("无有效边界框图像数量:", no_valid_bbox)

    print("越界边界框图像数量:", clipped)

    print("\n输出目录:", OUT_DIR)

    print("\n成功生成5个分析文件！")


if __name__ == "__main__":

    main()