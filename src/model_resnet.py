
import torch
import torch.nn as nn

from torchvision.models import resnet34
from torchvision.models import ResNet34_Weights


# ==========================================
# 1. ResNet34 编码器
# ==========================================

class ResNet34Encoder(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # 判断是否使用预训练权重
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None

        # 加载 ResNet34
        backbone = resnet34(weights=weights)

        # 第一阶段：初始卷积
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # 最大池化
        self.maxpool = backbone.maxpool

        # 四个残差阶段
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):

        # 第一阶段
        x = self.conv1(x)
        x = self.bn1(x)
        skip1 = self.relu(x)

        # 最大池化
        x = self.maxpool(skip1)

        # 第二阶段
        skip2 = self.layer1(x)

        # 第三阶段
        skip3 = self.layer2(skip2)

        # 第四阶段
        skip4 = self.layer3(skip3)

        # 最深层特征
        deep_feature = self.layer4(skip4)

        # 返回五个不同尺度的特征
        return (
            skip1,
            skip2,
            skip3,
            skip4,
            deep_feature
        )

# ==========================================
# 2. DoubleConv 模块
# ==========================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ==========================================
# 3. DecoderBlock 模块
# ==========================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        # 转置卷积：恢复空间尺寸
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # 拼接后进行卷积融合
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # 第一步：上采样
        x = self.up(x)

        # 检查空间尺寸
        if x.shape[2:] != skip.shape[2:]:

            raise ValueError(
                "上采样特征与 Skip 特征尺寸不一致"
            )

        # 第二步：通道拼接
        x = torch.cat(
            [x, skip],
            dim=1
        )

        # 第三步：卷积融合
        x = self.conv(x)

        return x
# ==========================================
# 2. 测试编码器
# ==========================================

# ==========================================
# 4. 完整的 Decoder
# ==========================================

class ResNet34Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        # 第一次解码：8 -> 16
        self.dec1 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        # 第二次解码：16 -> 32
        self.dec2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        # 第三次解码：32 -> 64
        self.dec3 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        # 第四次解码：64 -> 128
        self.dec4 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64
        )

        # 最后一次上采样：128 -> 256
        self.final_up = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )

        # 最终特征融合
        self.final_conv = DoubleConv(
            in_channels=32,
            out_channels=32
        )

    def forward(
        self,
        skip1,
        skip2,
        skip3,
        skip4,
        deep_feature
    ):

        # 第一次解码
        x = self.dec1(deep_feature, skip4)

        # 第二次解码
        x = self.dec2(x, skip3)

        # 第三次解码
        x = self.dec3(x, skip2)

        # 第四次解码
        x = self.dec4(x, skip1)

        # 最后一次上采样
        x = self.final_up(x)

        # 最终特征融合
        x = self.final_conv(x)

        return x


# ==========================================
# 5. 输出层
# ==========================================

class OutputLayer(nn.Module):

    def __init__(self, in_channels=32, out_channels=1):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        return self.conv(x)


# ==========================================
# 6. 完整的 ResNet34-U-Net
# ==========================================

class ResNet34UNet(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # 编码器
        self.encoder = ResNet34Encoder(
            pretrained=pretrained
        )

        # 解码器
        self.decoder = ResNet34Decoder()

        # 输出层
        self.output_layer = OutputLayer(
            in_channels=32,
            out_channels=1
        )

    def forward(self, x):

        # 第一步：提取多尺度特征
        skip1, skip2, skip3, skip4, deep_feature = \
            self.encoder(x)

        # 第二步：解码并恢复空间尺寸
        x = self.decoder(
            skip1,
            skip2,
            skip3,
            skip4,
            deep_feature
        )

        # 第三步：生成单通道 Logits
        x = self.output_layer(x)

        return x

# ==========================================
# 5. 测试 Encoder 和 Decoder
# ==========================================


# ==========================================
# 7. 测试完整网络
# ==========================================

if __name__ == "__main__":

    from losses import CombinedLoss

    # 1. 设置设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备：", device)

    # 2. 创建完整模型
    model = ResNet34UNet(
        pretrained=True
    ).to(device)

    model.train()

    # 3. 模拟输入图像
    images = torch.rand(
        2, 3, 256, 256
    ).to(device)

    # 4. 模拟二值 Mask
    masks = torch.randint(
        0, 2, (2, 1, 256, 256)
    ).float().to(device)

    # 5. 前向传播
    prediction = model(images)

    print("\n========== 形状检查 ==========")

    print("Images：", images.shape)
    print("Masks：", masks.shape)
    print("Prediction：", prediction.shape)

    assert prediction.shape == masks.shape

    # 6. 检查参数量
    encoder_params = sum(
        p.numel() for p in model.encoder.parameters()
    )

    decoder_params = sum(
        p.numel() for p in model.decoder.parameters()
    )

    output_params = sum(
        p.numel() for p in model.output_layer.parameters()
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("\n========== 参数量检查 ==========")

    print("Encoder 参数量：", encoder_params)
    print("Decoder 参数量：", decoder_params)
    print("OutputLayer 参数量：", output_params)
    print("模型总参数量：", total_params)

    # 7. 计算 Loss
    criterion = CombinedLoss()

    loss = criterion(prediction, masks)

    print("\n========== Loss 检查 ==========")

    print("Loss：", loss.item())

    # 8. 反向传播
    model.zero_grad()
    loss.backward()

    print("\n========== 梯度检查 ==========")

    print(
        "Encoder 是否获得梯度：",
        model.encoder.conv1.weight.grad is not None
    )

    print(
        "OutputLayer 是否获得梯度：",
        model.output_layer.conv.weight.grad is not None
    )

    print("\n完整模型测试结束！")
