
import torch
import torch.nn as nn

from attention import AttentionGate


# ==================================
# 1. DoubleConv：双卷积模块
# ==================================

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


# ==================================
# 2. EncoderBlock：编码器基本模块
# ==================================

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 双卷积
        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

        # 最大池化
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        # 提取特征
        features = self.conv(x)

        # 下采样
        downsampled = self.pool(features)

        return features, downsampled


# ==================================
# 3. Encoder：完整编码器
# ==================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = EncoderBlock(3, 64)

        self.enc2 = EncoderBlock(64, 128)

        self.enc3 = EncoderBlock(128, 256)

        self.enc4 = EncoderBlock(256, 512)

    def forward(self, x):

        skip1, x = self.enc1(x)

        skip2, x = self.enc2(x)

        skip3, x = self.enc3(x)

        skip4, x = self.enc4(x)

        return skip1, skip2, skip3, skip4, x


# ==================================
# 4. Bottleneck：瓶颈层
# ==================================

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x):
        return self.conv(x)


# ==================================
# 5. UpSample：转置卷积上采样
# ==================================

class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        return self.up(x)


# ==================================
# 6. DecoderBlock：注意力解码器模块
# ==================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        # 1. 上采样
        self.up = UpSample(
            in_channels,
            out_channels
        )

        # 2. Attention Gate
        self.attention = AttentionGate(
            F_g=out_channels,
            F_l=skip_channels,
            F_int=out_channels // 2
        )

        # 3. 拼接后进行双卷积
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # 1. Decoder 特征上采样
        x = self.up(x)

        # 2. 检查空间尺寸
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                "Decoder 和 Encoder 特征图尺寸不一致！"
            )

        # 3. Attention Gate 筛选 Encoder 特征
        skip = self.attention(
            skip,
            x
        )

        # 4. 拼接筛选后的 Encoder 特征
        x = torch.cat(
            [x, skip],
            dim=1
        )

        # 5. 双卷积
        x = self.conv(x)

        return x


# ==================================
# 7. Decoder：完整解码器
# ==================================

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 第四层
        self.dec1 = DecoderBlock(
            1024, 512, 512
        )

        # 第三层
        self.dec2 = DecoderBlock(
            512, 256, 256
        )

        # 第二层
        self.dec3 = DecoderBlock(
            256, 128, 128
        )

        # 第一层
        self.dec4 = DecoderBlock(
            128, 64, 64
        )

    def forward(
        self,
        x,
        skip1,
        skip2,
        skip3,
        skip4
    ):

        # 从最深层开始逐步恢复
        x = self.dec1(x, skip4)

        x = self.dec2(x, skip3)

        x = self.dec3(x, skip2)

        x = self.dec4(x, skip1)

        return x


# ==================================
# 8. OutputLayer：分割输出层
# ==================================

class OutputLayer(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


# ==================================
# 9. Attention U-Net：完整网络
# ==================================

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # 1. 编码器
        self.encoder = Encoder()

        # 2. 瓶颈层
        self.bottleneck = Bottleneck(
            512, 1024
        )

        # 3. 注意力解码器
        self.decoder = Decoder()

        # 4. 输出层
        self.output_layer = OutputLayer(
            64, 1
        )

    def forward(self, x):

        # Encoder
        skip1, skip2, skip3, skip4, x = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder + Attention Gate
        x = self.decoder(
            x,
            skip1,
            skip2,
            skip3,
            skip4
        )

        # Output
        x = self.output_layer(x)

        return x


# ==================================
# 10. 完整网络测试
# ==================================

if __name__ == "__main__":

    # 创建 Attention U-Net
    model = UNet()

    # 切换到评估模式
    model.eval()

    # 模拟输入图像
    images = torch.randn(
        1, 3, 256, 256
    )

    print("========== Attention U-Net 测试 ==========")

    # 前向传播
    with torch.no_grad():

        prediction = model(images)

    # 输入输出形状
    print("输入形状：", images.shape)

    print("输出形状：", prediction.shape)

    # 参数量统计
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Attention Gate 数量
    attention_count = sum(
        1
        for module in model.modules()
        if isinstance(module, AttentionGate)
    )

    print("Attention Gate 数量：", attention_count)

    print("模型总参数量：", total_params)

    print("可训练参数量：", trainable_params)

    # 检查输出
    assert prediction.shape == (
        1, 1, 256, 256
    ), "模型输出尺寸错误！"

    print("模型前向传播测试通过！")
