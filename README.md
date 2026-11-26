# EndoPolyp-Seg · 结肠镜息肉分割

## 项目简介

这是一个用 PyTorch 做的结肠镜息肉分割项目。我用 Kvasir-SEG 数据集训练了基础 U-Net、ResNet34-U-Net 和 Attention U-Net，用同一份测试集比较效果，再用 Gradio 做了一个能上传图片、切换模型和并排查看结果的页面。

![结肠镜原图、人工标注和三个模型的预测掩码](outputs_comparison/visualizations/0002_cju8doa16u5gh0818w1ywda3q.jpg)

*从左到右是原图、人工标注、Original U-Net、ResNet34-U-Net 和 Attention U-Net 的预测。白色区域是息肉掩码。*

## 主要功能

- **单模型预测**可以任选一个模型，查看预测掩码、红色叠加图和前景面积占比。
- **三模型对照**用同一张图跑完三个模型，把掩码、叠加图和推理耗时放在一起看。
- **单图评估**支持上传对应的人工标注，计算这张图的 Dice 和 IoU。
- **批量评估**保存了 150 张测试图的逐图结果，以及 Batch=1、Batch=4 的 GPU 测速报告。

## 快速开始

演示在 Windows 10、Python 3.11、PyTorch 2.6.0、torchvision 0.21.0 和 Gradio 6.32.0 的环境中跑过。先按设备选择 CPU 或 CUDA 对应的 PyTorch 和 torchvision 安装包。[PyTorch 安装指南](https://pytorch.org/get-started/previous-versions/#v260)

```bash
git clone https://github.com/Drifter53/EndoPolyp-Seg.git
cd EndoPolyp-Seg
python -m pip install gradio==6.32.0 opencv-python Pillow numpy
```

### 准备模型权重

Gradio 启动时需要三个检查点。目前权重保存在本机，暂未提供公开下载。有权重后按下表放置，准备一张结肠镜图片即可演示。完整的数据集用于训练和评估。

| 模型 | 权重路径 |
| --- | --- |
| Original U-Net | `outputs/best_unet.pth` |
| ResNet34-U-Net | `outputs_resnet34/best_unet.pth` |
| Attention U-Net | `outputs_attention/best_attention_unet.pth` |

在项目根目录启动 Gradio。

```bash
python app.py
```

启动后打开本地页面。Windows 启动命令、演示用图和关闭方式见运行说明。[打开本地页面](http://127.0.0.1:7861) · [运行说明](RUNNING.md)

## 网页怎么用

1. 在“上传结肠镜图像”中选择一张图片。有对应的标注时，可以一起上传 Mask。
2. 选一个模型点“单模型预测”，或点“三模型一键对照”并切到“三模型对照”页签。
3. 看预测掩码和红色叠加图。上传了标注时，页面也会显示 Dice 和 IoU。换图前可以点“清空”。

## 实验结果

### 分割效果

三种模型用同一份 **150 张图像**的测试集，输入统一为 **256×256**。预测概率达到 0.5 的像素记为前景，Dice 和 IoU 按图计算后取平均。

| 模型 | 平均 Dice | 平均 IoU | 测试 Loss |
| --- | ---: | ---: | ---: |
| Original U-Net | 0.804851 | 0.718354 | 0.413240 |
| ResNet34-U-Net | **0.880280** | **0.812312** | 0.253274 |
| Attention U-Net | 0.816444 | 0.726136 | 0.406886 |

在这份测试集上，ResNet34-U-Net 的 Dice 和 IoU 都最高。我还按标注面积和位置给测试图分了组，查看不同大小和位置的息肉分得怎么样。[测试汇总](outputs_comparison/summary.csv) · [逐图结果](outputs_comparison/per_image_metrics.csv) · [困难样本分析](outputs_difficulty_analysis/difficulty_analysis_report.txt)

### GPU 推理速度

在 **GTX 1060 6GB** 上用 **FP32、256×256** 输入测速。下面是稳态端到端结果，包含读图、预处理、数据传输和模型前向。Batch=1 按单张图计时，Batch=4 按完整的四张一批计时。

| 模型 | Batch=1 单张延迟 | Batch=4 整批延迟 | Batch=4 吞吐量 |
| --- | ---: | ---: | ---: |
| Original U-Net | 30.338 ms/图 | 104.236 ms/批 | 38.37 图/秒 |
| ResNet34-U-Net | **16.755 ms/图** | **39.605 ms/批** | **101.00 图/秒** |
| Attention U-Net | 32.158 ms/图 | 112.525 ms/批 | 35.55 图/秒 |

网页上显示的是一次预热后五次模型前向的中位数。报告中还有纯前向耗时、参数量、计算量和显存峰值。[Batch=1 报告](outputs_benchmark_v4_batch1/benchmark_v4_batch1_report.txt) · [Batch=4 报告](outputs_benchmark_v3/benchmark_v3_report.txt)

## 技术实现

| 模型 | 改了什么 | 参数量 |
| --- | --- | ---: |
| [Original U-Net](src/model.py) | 四级编码器和解码器，上采样后拼接同尺度的编码特征 | 31.032 M |
| [ResNet34-U-Net](src/model_resnet.py) | 用 ImageNet 预训练的 ResNet34 作编码器，再接 U-Net 解码器 | 24.450 M |
| [Attention U-Net](src/model_attention.py) | 在四处跳跃连接加入 Attention Gate，先给编码特征加权再拼接 | 31.382 M |

训练、验证、测试分别用了 **699、150、150** 对图像和标注。图像缩放时用双线性插值，掩码用最近邻插值。训练时对图像和标注一起做随机翻转和旋转。基础 U-Net 的 RGB 输入缩放到 0–1，其余两个模型再做 ImageNet 标准化。[划分文件](data/splits) · [预处理](src/dataset.py) · [数据增强](src/transforms.py)

训练用 Adam 和 BCE + Dice Loss，按验证集 Loss 选最佳权重。基础 U-Net 和 Attention U-Net 的学习率为 `1e-4`。ResNet34-U-Net 给预训练编码器设 `1e-5`，解码器和输出层用 `1e-4`。[Original 训练日志](outputs/training_history.csv) · [ResNet34 训练日志](outputs_resnet34/training_history.csv) · [Attention 训练日志](outputs_attention/attention_training_history.csv)

<details>
<summary>重跑训练和评估</summary>

从数据集官网下载 Kvasir-SEG，将图像和掩码分别放进 `data/Kvasir-SEG/images/` 和 `data/Kvasir-SEG/masks/`，沿用仓库里 `data/splits/` 的划分文件。[Kvasir-SEG 官方下载](https://datasets.simula.no/kvasir-seg/)

从项目根目录执行。训练和评估会向对应的输出目录写入文件。测速需要 CUDA，计算 MACs 和 FLOPs 时需要安装 `thop`。

```bash
# 训练三种模型，按需运行
python src/train.py
python src/train_resnet.py
python src/train_attention.py

# 加载相应权重，跑三模型对比
python src/compare_models.py

# 分别跑 Batch=1 和 Batch=4
python src/benchmark_v4_batch1.py
python src/benchmark_v3.py
```

</details>

## 项目结构

```text
EndoPolyp-Seg/
├── app.py                          # Gradio 界面和推理入口
├── RUNNING.md                      # 启动步骤和演示样例
├── data/splits/                    # 训练、验证、测试名单
├── src/
│   ├── model*.py                   # 三种模型
│   ├── dataset.py                  # 数据读取和预处理
│   ├── train*.py                   # 模型训练
│   ├── compare_models.py           # 三模型测试对比
│   ├── benchmark_v4_batch1.py      # Batch=1 测速
│   └── benchmark_v3.py             # Batch=4 测速
├── outputs_comparison/             # 测试汇总、逐图指标和预测图
├── outputs_benchmark_v4_batch1/    # Batch=1 测速报告
├── outputs_benchmark_v3/           # Batch=4 测速报告
└── outputs_difficulty_analysis/    # 按息肉大小和位置分组的结果
```

## 常见问题

| 遇到的问题 | 怎么处理 |
| --- | --- |
| 启动时提示找不到权重 | 核对三个 `.pth` 的文件名和目录，程序会在启动时一起加载。 |
| 浏览器打不开页面 | 确认终端已经显示本地 URL，再访问 `127.0.0.1:7861`。如果端口被已有服务占用，先在旧服务的终端按 `Ctrl+C`。 |
| 页面没显示 Dice 和 IoU | 上传与原图对应的人工标注 Mask，再点一次预测。 |

## 数据来源

Debesh Jha 等，*Kvasir-SEG: A Segmented Polyp Dataset*，MMM 2020。[数据集论文](https://arxiv.org/abs/1911.07069)
