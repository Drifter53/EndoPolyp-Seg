# EndoPolyp-Seg

基于 PyTorch 的结肠镜息肉二值分割项目，围绕 Kvasir-SEG 完成数据处理、三种 U-Net 模型训练、测试集评价、GPU 性能分析和 Gradio 交互展示。项目面向计算机视觉实习求职与现场演示，重点展示完整流程及可追溯的实验结果。

在现有 150 张测试图像上，ResNet34-U-Net 的平均 Dice 为 **0.880280**、平均 IoU 为 **0.812312**；在 GTX 1060 6GB、FP32、256×256 输入下，Batch=1 稳态端到端平均耗时为 **16.755 ms/图**。精度来自三模型对照报告，速度来自独立的 Batch=1 benchmark，具体口径见下文。

快速启动与现场演示步骤见 [项目运行说明](RUNNING.md)。

项目地址：[Drifter53/EndoPolyp-Seg](https://github.com/Drifter53/EndoPolyp-Seg)。仓库包含代码、文档、数据划分和实验报告；原始数据集、模型权重、缓存及部分历史实验未上传，仍保留在本机。其他电脑运行前需按下列目录补齐数据和权重，目前尚未提供下载链接。

## 1. 已实现功能

- Kvasir-SEG 图像与 Mask 配对、固定划分文件、训练集同步数据增强。
- Original U-Net、ResNet34-U-Net、Attention U-Net 三种模型训练、验证、检查点保存与恢复。
- 测试集逐图 Dice / IoU、结果可视化及三模型对照报告。
- Batch=1 与 Batch=4 的独立 GPU benchmark，记录前向耗时、端到端耗时、吞吐量、参数量、MACs / FLOPs 和显存峰值。
- 基于现有测试结果的息肉尺寸、位置分组与困难样本分析。
- Gradio 图片上传、单模型预测、三模型一键对照、二值 Mask 与红色叠加图；可选上传人工标注计算该图的 Dice / IoU，并显示设备、耗时和预测前景占比。

## 2. 三种模型与权重

三个模型均输出单通道 logits，推理时使用 sigmoid 和 0.5 阈值生成前景 Mask。

| 模型              | 实际实现                                                                                                                         | 输入归一化                     |        参数量 |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ---------: |
| Original U-Net  | [model.py](src/model.py) 的 `UNet`；四级编码器通道为 64 / 128 / 256 / 512，瓶颈为 1024；双 3×3 卷积与 ReLU，最大池化下采样，转置卷积上采样、跳跃连接拼接               | RGB / 255，不做 ImageNet 标准化 | 31,031,745 |
| ResNet34-U-Net  | [model\_resnet.py](src/model_resnet.py) 的 `ResNet34UNet`；torchvision ResNet34 编码器、多尺度跳跃连接、自定义 U-Net 解码器，最后额外上采样恢复输入尺寸        | RGB / 255，再做 ImageNet 标准化 | 24,450,497 |
| Attention U-Net | [model\_attention.py](src/model_attention.py) 的 `UNet`；沿用四级 U-Net 主干，在四处跳跃连接加入 [AttentionGate](src/attention.py)，对编码器特征加权后拼接 | RGB / 255，再做 ImageNet 标准化 | 31,382,309 |

ResNet34 首次训练使用 `ResNet34_Weights.DEFAULT`；推理、测试和 benchmark 使用 `pretrained=False` 建模后加载项目完整权重，不需再次下载 ImageNet 权重。Original U-Net 与 Attention U-Net 从随机初始化开始训练。

### 当前演示与正式对比使用的检查点

| 模型              | 权重路径                                                                                      | 最佳检查点 epoch |  验证 Loss |
| --------------- | ----------------------------------------------------------------------------------------- | ----------: | -------: |
| Original U-Net  | `outputs/best_unet.pth`                                           |          84 | 0.381007 |
| ResNet34-U-Net  | `outputs_resnet34/best_unet.pth`                        |          28 | 0.253735 |
| Attention U-Net | `outputs_attention/best_attention_unet.pth` |          89 | 0.390904 |

这三个检查点已与对应模型执行 `strict=True` 参数加载核对，全部匹配。表中验证 Loss 是检查点元数据，**不是测试集结果**。Original U-Net 旧检查点没有保存模型名、输入尺寸和归一化标志；本文按当前训练、测试、对比与演示代码共同使用的 256×256、`normalize_imagenet=False` 约定说明，不声称旧检查点具备完整训练配置记录。

## 3. 数据与预处理

本地 `data/Kvasir-SEG/images/` 和 `masks/` 各有 1000 个图像文件，按文件名配对。当前实验使用 [data/splits](data/splits) 中已有的文件，每行以制表符分隔图像名和 Mask 名。

| 划分  | 当前有效配对数 | 文件                                 |
| --- | ------: | ---------------------------------- |
| 训练集 |     699 | [train.txt](data/splits/train.txt) |
| 验证集 |     150 | [val.txt](data/splits/val.txt)     |
| 测试集 |     150 | [test.txt](data/splits/test.txt)   |

三个划分没有重叠，列出的配对文件均存在。当前合计为 **999 对**，不是 1000 对：`cju88l66no10s0850rsda7ej1.jpg` 未出现在任何划分中，其排除原因没有找到记录，**需用户确认**。现有 [split\_dataset.py](src/split_dataset.py) 对全量配对排序后以随机种子 42 打乱，按 70% / 15% / 15% 重新写入划分；直接执行会改变现有划分，因此复用当前结果时应保留已有三个 txt 文件。

实际预处理位于 [dataset.py](src/dataset.py) 和 [transforms.py](src/transforms.py)：

1. OpenCV 读取图像并将 BGR 转为 RGB；Mask 以灰度读取，同时检查原图与 Mask 尺寸一致。
2. 图像用双线性插值缩放至 256×256；Mask 用最近邻插值缩放，按 `mask > 127` 转成 0 / 1。
3. 图像转为 float32、CHW 格式并除以 255；Mask 为 `1×256×256`。
4. 仅训练集启用同步增强：水平翻转与垂直翻转各以 0.5 概率发生，并随机选择 0° / 90° / 180° / 270° 旋转。验证集和测试集不做增强。
5. ResNet34-U-Net 和 Attention U-Net 额外使用 ImageNet 均值 `[0.485, 0.456, 0.406]`、标准差 `[0.229, 0.224, 0.225]`；Original U-Net 保持 `[0, 1]` 输入。

## 4. 训练与评价流程

### 训练设置与记录

三个训练入口都使用 Adam 和 [CombinedLoss](src/losses.py)：`BCEWithLogitsLoss + DiceLoss`，默认权重均为 1；其中 DiceLoss 使用概率预测，平滑项为 1。每轮训练后在验证集评价，以更低的验证 Loss 保存最佳权重，同时保存最新检查点与 CSV 日志。

| 模型              | 训练入口                                          | Batch / workers | 学习率                   |  当前代码目标 epoch | 现有日志最后 epoch |
| --------------- | --------------------------------------------- | --------------- | --------------------- | ------------: | -----------: |
| Original U-Net  | [train.py](src/train.py)                      | 默认 4 / 0        | 全部参数 1e-4             | 默认 100，可用参数指定 |          100 |
| ResNet34-U-Net  | [train\_resnet.py](src/train_resnet.py)       | 4 / 0           | 编码器 1e-5；解码器及输出层 1e-4 |           101 |          101 |
| Attention U-Net | [train\_attention.py](src/train_attention.py) | 4 / 0           | 全部参数 1e-4             |           100 |          101 |

日志分别为 [Original](outputs/training_history.csv)、[ResNet34](outputs_resnet34/training_history.csv)、[Attention](outputs_attention/attention_training_history.csv)。Attention 日志记录至 101，但当前脚本设为 100；当时使用的轮数配置**需用户确认**，本文不将三次训练统一描述为相同轮数。

训练脚本会检测各自输出目录的最新检查点并恢复模型、优化器和轮数。Original U-Net 支持命令行参数；另两个训练入口的配置保存在脚本常量中，没有等价的 `--epochs` 或 `--output-dir` 参数。

### 测试与指标口径

独立测试入口为 [test.py](src/test.py)、[test\_resnet.py](src/test_resnet.py)、[test\_attention.py](src/test_attention.py)。三模型统一对照入口为 [compare\_models.py](src/compare_models.py)，使用相同的 150 张测试图像，输入为 256×256，按各模型的归一化约定处理。

正式对照与 benchmark 先将 `sigmoid(logits) >= 0.5` 二值化，再逐图计算 Dice 和 IoU，平滑项为 `1e-7`，最后对全部测试图像取平均：

- `Dice = (2 × intersection + ε) / (pred_area + gt_area + ε)`
- `IoU = (intersection + ε) / (pred_area + gt_area - intersection + ε)`

注意：[metrics.py](src/metrics.py) 中供 ResNet34 / Attention 训练验证使用的实现为 `> 0.5`、平滑项 1；与正式测试口径存在差异。训练验证曲线、最佳检查点元数据、下表测试均值以及网页单图指标应分别理解。

## 5. 已有性能结果

### 5.1 测试集分割精度

来源：[outputs\_comparison/summary.csv](outputs_comparison/summary.csv) 和 [summary.txt](outputs_comparison/summary.txt)。以下是全部 150 张测试图像的结果，V3 / V4 benchmark 的精度与其一致（报告中显示为四位小数）。

| 模型              | Test Loss |    Mean Dice |     Mean IoU |
| --------------- | --------: | -----------: | -----------: |
| Original U-Net  |  0.413240 |     0.804851 |     0.718354 |
| ResNet34-U-Net  |  0.253274 | **0.880280** | **0.812312** |
| Attention U-Net |  0.406886 |     0.816444 |     0.726136 |

在本项目的 150 张测试图像上，ResNet34-U-Net 的平均 Dice 和 IoU 均为三种模型中最高。逐图评价结果和预测可视化见 [outputs_comparison](outputs_comparison)。

### 5.2 Benchmark 公共条件

两份报告均使用 NVIDIA GeForce GTX 1060 6GB、PyTorch 2.6.0+cu124、CUDA runtime 12.4、FP32、256×256 输入，`num_workers=0`、`cuDNN benchmark=True`。

- **GPU model-only**：图像传入 GPU 并完成标准化后，仅对模型 forward 计时；预热 20 次，正式计时 100 次。
- **端到端（E2E）**：计时包含 DataLoader 读取与 CPU 预处理、图像 CPU→GPU 传输、GPU 标准化、模型 forward 和 CUDA 同步。Mask 保留在 CPU，指标计算不计时；也不包含网页上传、模型加载与设备搬移、Mask 恢复尺寸和页面展示。
- 正式计时前对实际 batch shape 分别预热 5 次，E2E 另预热 3 个 batch。稳态统计排除正式遍历首批；不完整尾批单独报告，所有 150 张图仍参与精度统计。
- 表中耗时均取各报告的 **Mean**，吞吐量单位为 images/s。显存使用报告中的 PyTorch peak allocated / reserved，单位为 GiB；它是对应脚本运行过程的记录，不等于检查点大小或部署最低显存要求。

### 5.3 Batch=1：单张图像延迟

来源：[V4 Batch=1 报告](outputs_benchmark_v4_batch1/benchmark_v4_batch1_report.txt) 与 [汇总 CSV](outputs_benchmark_v4_batch1/benchmark_v4_batch1.csv)。150 个 batch 无不完整尾批，稳态统计为排除首批后的 149 张。

| 模型              | GPU forward 平均 ms/图 | 稳态 E2E 平均 ms/图 | 稳态 E2E images/s | 完整遍历 images/s | 峰值 allocated / reserved GiB |
| --------------- | ------------------: | -------------: | --------------: | ------------: | --------------------------: |
| Original U-Net  |              24.199 |         30.338 |           32.96 |         32.96 |               0.219 / 0.262 |
| ResNet34-U-Net  |              10.603 |         16.755 |           59.68 |         59.63 |               0.126 / 0.158 |
| Attention U-Net |              26.846 |         32.158 |           31.10 |         31.11 |               0.330 / 0.354 |

### 5.4 Batch=4：批量吞吐量

来源：[V3 Batch=4 报告](outputs_benchmark_v3/benchmark_v3_report.txt) 与 [汇总 CSV](outputs_benchmark_v3/benchmark_v3_batch4.csv)。150 张组成 37 个完整 batch 和 1 个 Batch=2 尾批；排除首批与尾批后，稳态为 36 个完整 batch，共 144 张。

| 模型              | GPU forward 平均 ms/batch | 稳态 E2E 平均 ms/batch | 稳态 E2E images/s | 完整遍历 images/s | 峰值 allocated / reserved GiB |
| --------------- | ----------------------: | -----------------: | --------------: | ------------: | --------------------------: |
| Original U-Net  |                  88.581 |            104.236 |           38.37 |         38.39 |               4.786 / 4.932 |
| ResNet34-U-Net  |                  24.405 |             39.605 |          101.00 |        100.97 |               4.453 / 4.471 |
| Attention U-Net |                  97.936 |            112.525 |           35.55 |         35.54 |               1.596 / 1.789 |

**Batch=4 的 ms/batch 表示整批四张的耗时；即使除以 4，也只是批处理摊销值，不能代替 Batch=1 单张请求延迟。** 完整遍历吞吐量包含首批和尾批，稳态吞吐量不包含它们。早期 `outputs_benchmark/` 和 `outputs_benchmark_v2/` 保留为历史记录，本文不将其速度混入 V3 / V4 主结果。

### 5.5 计算量

来源同 V3 / V4，按单张 256×256 输入统计；FLOPs 使用 `2 × MACs` 的报告口径。

| 模型              |  参数量 M | MACs G/图 | FLOPs G/图 |
| --------------- | -----: | -------: | --------: |
| Original U-Net  | 31.032 |   54.610 |   109.220 |
| ResNet34-U-Net  | 24.450 |   10.393 |    20.786 |
| Attention U-Net | 31.382 |   55.719 |   111.438 |

### 5.6 已有困难样本分析

[outputs\_difficulty\_analysis](outputs_difficulty_analysis) 基于现有测试集逐图结果，按 GT Mask 的前景占比分为 Small / Medium / Large，各 50 张；按归一化质心划分 Center 124 张、Peripheral 26 张。定义和分组精度见 [difficulty\_analysis\_report.txt](outputs_difficulty_analysis/difficulty_analysis_report.txt)，逐图结果见 [per\_image\_model\_metrics.csv](outputs_difficulty_analysis/per_image_model_metrics.csv)。这些分组是当前测试集上的分析，不是新增数据集或额外训练结果。

## 6. 环境配置

### 当前本机已核对的环境

核对日期：2026-11-26。实际解释器为 `C:\Users\23180\Desktop\PythonProject\.venv-gpu311\Scripts\python.exe`，系统为 Windows 10，Python 为 3.11.18，CUDA 可用，显卡为 GTX 1060 6GB。

| 依赖                    | 当前安装版本                     |
| --------------------- | -------------------------- |
| torch / torchvision   | 2.6.0+cu124 / 0.21.0+cu124 |
| gradio                | 6.32.0                     |
| numpy / opencv-python | 2.4.6 / 5.1.0.95           |
| Pillow                | 13.0.0                     |
| matplotlib / pandas   | 3.11.2 / 3.1.1             |
| thop                  | 0.1.1-2209072238           |

仓库当前没有 `requirements.txt`、环境锁文件或权重下载脚本。以上为当前安装状态；历史 benchmark 报告只记录了 GPU、PyTorch 和 CUDA 等部分信息，不能据此声称所有历史实验使用了表内全部版本。

### 使用现有环境

以下命令使用 **PowerShell**。先进入项目根目录；后续训练、测试和演示命令均以该目录为工作目录。当前本机目录对应：

```powershell
Set-Location 'C:\Users\23180\Desktop\PythonProject\CV-Internship\EndoPolyp-Seg'
$python = (Resolve-Path '..\..\.venv-gpu311\Scripts\python.exe').Path
& $python -B -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

### 新机器环境配置参考

先准备 Python 3.11，再在项目根目录执行。以下安装命令按本机已安装版本整理，**本轮未执行全新环境安装，跨机器完整复现需确认**；GPU 是否可用还取决于目标机器的 NVIDIA 驱动与硬件。

```powershell
py -3.11 -m venv .venv
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
& $python -m pip install gradio==6.32.0 numpy==2.4.6 opencv-python==5.1.0.95 Pillow==13.0.0 matplotlib==3.11.2 pandas==3.1.1 thop==0.1.1-2209072238
```

数据和权重需要另行放入本文列出的实际目录。尚无已确认的对外下载地址，**需用户确认发布方式**；本文不提供虚构的仓库地址、数据或 checkpoint 下载命令。

## 7. Gradio 演示

### 启动

确认 `$python` 已按上一节设置，且第 2 节的三个最佳权重均在原位置。仅演示自选图片时不需要本地完整训练数据集。

```powershell
& $python -B app.py
```

入口为根目录的 [app.py](app.py)，程序启动时先在 CPU 加载三个模型，再调用 `demo.queue().launch(...)`。服务固定监听 `127.0.0.1:7861`，并设置 `inbrowser=True` 自动打开浏览器；也可手动访问 [本地演示页面](http://127.0.0.1:7861)。终端按 `Ctrl+C` 停止服务。端口若被占用，应先结束占用该端口的已有服务；当前代码没有命令行端口参数。

### 操作步骤

1. 上传一张结肠镜图像。
2. 可选上传与图像对应的人工标注 Mask：前景白、背景黑。也可使用 `data/Kvasir-SEG/images/` 与 `masks/` 中同名文件演示。
3. 选择模型并点击“单模型预测”，在对应页签查看原图、GT（如有）、预测 Mask、叠加图、耗时及前景占比。默认模型为 ResNet34-U-Net。
4. 点击“三模型一键对照”，在“三模型对照”页签查看三组 Mask、叠加图和对照表。
5. 点击“清空”重置输入与结果。

应用自动选择 CUDA 或 CPU。三模型对照按顺序运行，任一时刻仅将当前模型搬入推理设备，结束后移回 CPU；因此按钮点击至结果显示的总等待时间还包含模型搬移与多次前向传播。

网页图像先缩放至 256×256 推理，二值化后用最近邻插值恢复原图尺寸。上传的 GT 会转换为灰度、必要时调整至原图大小并二值化。网页 Dice / IoU 在**恢复后的原图尺寸**上计算；未上传 GT 时不计算精度，因此网页单图分数不能直接当作测试集平均结果。

网页耗时使用 **1 次预热后 5 次前向传播的中位数**，不含预处理、模型及输入设备传输、后处理和页面展示。单模型结果中的“平均推理耗时”标签实际对应此中位数；它与第 5 节的 benchmark 均值及 E2E 延迟口径不同。

2026-11-26 已在现有 CUDA 环境完成服务启动、浏览器访问、图片及标注上传、单模型预测、三模型对照、模型切换、空输入提示和清空操作的实测，具体步骤见 [项目运行说明](RUNNING.md)。当前 Gradio 6.32.0 会提示 `theme` 参数应移至 `launch()`，本机服务启动与预测均成功；此次未修改源代码。核验结束后服务已停止。


## 8. 现有训练、评估与测速入口

以下是现有脚本的真实调用方式，供理解与后续复现使用；**整理 README 时未执行这些训练、评估或测速命令**。它们会写入或更新对应实验输出。若只需面试演示，运行上一节的 `app.py` 即可。

### 训练

```powershell
& $python -B src/train.py
& $python -B src/train_resnet.py
& $python -B src/train_attention.py
```

Original U-Net 默认输出至 `outputs/`，ResNet34 至 `outputs_resnet34/`，Attention 至 `outputs_attention/`；自动续训规则见第 4 节。当前已有完整训练记录，不需要为运行演示重新训练。

### 独立测试与三模型对照

```powershell
& $python -B src/test.py
& $python -B src/test_resnet.py
& $python -B src/test_attention.py
& $python -B src/compare_models.py
```

当前 `test.py` 默认写入 `outputs/test_results_original_unet/`；另两个分别写入 `outputs_resnet34/test_results/` 和 `outputs_attention/test_results/`。统一对照写入 `outputs_comparison/`，包含汇总、150 张逐图指标和前 20 张可视化。

### GPU Benchmark

```powershell
& $python -B src/benchmark_v4_batch1.py
& $python -B src/benchmark_v3.py
```

前者固定 Batch=1，输出至 `outputs_benchmark_v4_batch1/`；后者固定 Batch=4，输出至 `outputs_benchmark_v3/`，均包含汇总 CSV、逐图 CSV、逐批 CSV 和文本报告。两个入口明确要求 CUDA，不会自动退回 CPU；计算量统计使用 `thop`。它们会重写各自已有报告，因此阅读现有结果无需再次执行。

## 9. 目录结构

下面列出与主流程相关的现有文件和目录，省略缓存、部分教学辅助脚本及图片文件名。

```text
EndoPolyp-Seg/
├── .gitignore                       # 排除数据、权重与缓存
├── README.md
├── RUNNING.md                       # 启动、演示步骤与常见问题
├── app.py                           # Gradio 演示入口
├── data/
│   ├── Kvasir-SEG/
│   │   ├── images/                  # 1000 张图像
│   │   ├── masks/                   # 1000 张 Mask
│   │   └── kavsir_bboxes.json        # 原文件名，困难分析中辅助使用
│   └── splits/
│       ├── train.txt                # 当前 699 对
│       ├── val.txt                  # 150 对
│       └── test.txt                 # 150 对
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── split_dataset.py
│   ├── check_data_integrity.py
│   ├── model.py                     # Original U-Net
│   ├── model_resnet.py              # ResNet34-U-Net
│   ├── model_attention.py           # Attention U-Net
│   ├── attention.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── train_resnet.py
│   ├── train_attention.py
│   ├── test.py
│   ├── test_resnet.py
│   ├── test_attention.py
│   ├── compare_models.py
│   ├── benchmark.py                 # 早期版本
│   ├── benchmark_v2.py              # 历史版本
│   ├── benchmark_v3.py              # Batch=4 主报告入口
│   ├── benchmark_v4_batch1.py        # Batch=1 主报告入口
│   └── analyze_polyp_difficulty.py
├── outputs/
│   ├── best_unet.pth
│   ├── latest_checkpoint.pth
│   ├── training_history.csv
│   ├── data_integrity/
│   ├── test_results_original_unet/
│   ├── test_results/                # 已有其他测试输出，保留原状
│   └── old_run_50epoch/             # 历史记录
├── outputs_resnet34/
│   ├── best_unet.pth
│   ├── latest_checkpoint.pth
│   ├── training_history.csv
│   └── test_results/
├── outputs_attention/
│   ├── best_attention_unet.pth
│   ├── latest_attention_checkpoint.pth
│   ├── attention_training_history.csv
│   └── test_results/
├── outputs_comparison/
│   ├── summary.csv
│   ├── summary.txt
│   ├── per_image_metrics.csv
│   └── visualizations/
├── outputs_benchmark/               # 历史报告
├── outputs_benchmark_v2/            # 历史报告
├── outputs_benchmark_v3/            # Batch=4 报告与 CSV
├── outputs_benchmark_v4_batch1/     # Batch=1 报告与 CSV
├── outputs_difficulty_analysis/     # 尺寸、位置、逐图分析
└── outputs_352/                     # 历史实验，未混入本文主对比
```

## 10. 尚需确认的信息

- 当前训练划分少一张的原因，以及是否有对应的历史清理记录。
- Attention U-Net 训练日志至 101 轮、当前脚本目标为 100 轮的历史配置差异。
- 数据与权重的对外获取方式及发布许可尚需确认；公开代码仓库地址见文首，原始数据和权重不随仓库上传。
- 新机器的全新环境安装与 CPU 推理尚未单独实测；当前本机 CUDA 网页演示已核验，操作说明见 [RUNNING.md](RUNNING.md)。
