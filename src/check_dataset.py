
from pathlib import Path

# 数据集所在位置
root = Path(
    r"C:\Users\23180\Desktop\PythonProject\CV-Internship\EndoPolyp-Seg\data\Kvasir-SEG"
)

image_dir = root / "images"
mask_dir = root / "masks"

# 获取两个文件夹中的文件
image_files = [
    p for p in image_dir.iterdir() if p.is_file()
]

mask_files = [
    p for p in mask_dir.iterdir() if p.is_file()
]

# 提取文件名（不包含扩展名）
image_names = {p.stem for p in image_files}
mask_names = {p.stem for p in mask_files}

# 检查是否存在不匹配
missing_masks = image_names - mask_names
missing_images = mask_names - image_names

print("图像数量：", len(image_files))
print("Mask数量：", len(mask_files))

print("缺少Mask的图像数量：", len(missing_masks))
print("缺少图像的Mask数量：", len(missing_images))

if len(image_files) == 1000 and len(mask_files) == 1000 and image_names == mask_names:
    print("检查通过：图像与Mask一一对应！")
else:
    print("检查未完全通过，请检查数据集！")
