
from pathlib import Path
import random

# 1. 获取项目根目录
project_dir = Path(__file__).resolve().parents[1]

# 2. 设置图像和Mask目录
dataset_dir = project_dir / "data" / "Kvasir-SEG"

image_dir = dataset_dir / "images"
mask_dir = dataset_dir / "masks"

# 3. 读取图像和Mask文件
image_files = sorted([
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])
# 检查图像文件是否存在重复 stem
image_stems = [p.stem for p in image_files]

if len(image_stems) != len(set(image_stems)):
    raise ValueError("图像文件夹中存在重复文件名！")

# 检查 Mask 文件是否存在重复 stem
mask_stems = [
    p.stem for p in mask_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

if len(mask_stems) != len(set(mask_stems)):
    raise ValueError("Mask 文件夹中存在重复文件名！")
mask_files = {
    p.stem: p
    for p in mask_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

# 4. 建立图像和Mask配对
pairs = []

for image_path in image_files:
    mask_path = mask_files.get(image_path.stem)

    if mask_path is None:
        raise FileNotFoundError(
            f"找不到对应Mask：{image_path.name}"
        )

    pairs.append((image_path, mask_path))

# 5. 检查是否有多余Mask
image_names = {p.stem for p in image_files}
extra_masks = set(mask_files) - image_names

if extra_masks:
    raise ValueError(f"存在多余Mask：{extra_masks}")

# 6. 固定随机种子并打乱数据
random.Random(42).shuffle(pairs)

# 7. 按照70%、15%、15%划分
total = len(pairs)

train_num = int(total * 0.7)
val_num = int(total * 0.15)

train_pairs = pairs[:train_num]

val_pairs = pairs[
    train_num:train_num + val_num
]

test_pairs = pairs[
    train_num + val_num:
]

# 8. 创建划分结果目录
split_dir = project_dir / "data" / "splits"
split_dir.mkdir(parents=True, exist_ok=True)

# 9. 保存划分结果
def save_split(name, data):
    file_path = split_dir / f"{name}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        for image_path, mask_path in data:
            f.write(
                f"{image_path.name}\t{mask_path.name}\n"
            )

    print(f"{name}数量：{len(data)}")

save_split("train", train_pairs)
save_split("val", val_pairs)
save_split("test", test_pairs)

print("数据集划分完成！")
print("总数量：", total)
print("保存位置：", split_dir)
