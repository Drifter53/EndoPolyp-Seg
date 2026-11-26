
import torch
import torch.nn as nn

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

        # 第一层
        skip1, x = self.enc1(x)

        # 第二层
        skip2, x = self.enc2(x)

        # 第三层
        skip3, x = self.enc3(x)

        # 第四层
        skip4, x = self.enc4(x)

        return skip1, skip2, skip3, skip4, x

# ==================================
# Bottleneck：瓶颈层
# ==================================
class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 使用双卷积提取深层特征
        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x):
        return self.conv(x)

# ==================================
# UpSample：转置卷积上采样
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
# DecoderBlock：解码器模块
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

        # 2. 拼接后进行双卷积
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # 上采样
        x = self.up(x)

        # 检查空间尺寸是否一致
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError("特征图尺寸不一致！")

        # 拼接 Encoder 特征
        x = torch.cat(
            [x, skip],
            dim=1
        )

        # 双卷积
        x = self.conv(x)

        return x

# ==================================
# Decoder：完整解码器
# ==================================
class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 第四层：512 -> 256
        self.dec1 = DecoderBlock(
            1024, 512, 512
        )

        # 第三层：256 -> 128
        self.dec2 = DecoderBlock(
            512, 256, 256
        )

        # 第二层：128 -> 64
        self.dec3 = DecoderBlock(
            256, 128, 128
        )

        # 第一层：64 -> 64
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
# OutputLayer：分割输出层
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
# UNet：完整 U-Net 网络
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

        # 3. 解码器
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

        # Decoder
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
# 3. 测试代码
# ==================================
if __name__ == "__main__":

    # 模拟 DataLoader 输出的 Batch
    images = torch.randn(4, 3, 256, 256)

    # 测试 DoubleConv
    model = DoubleConv(3, 64)

    output = model(images)

    print("输入形状：", images.shape)
    print("DoubleConv输出形状：", output.shape)

    # 测试 MaxPool2d
    pool = nn.MaxPool2d(
        kernel_size=2,
        stride=2
    )

    pooled_output = pool(output)

    print("池化后形状：", pooled_output.shape)

    # 测试 EncoderBlock
    encoder = EncoderBlock(3, 64)

    features, downsampled = encoder(images)

    print("Encoder保留特征：", features.shape)
    print("Encoder下采样结果：", downsampled.shape)

    # 测试完整 Encoder
    full_encoder = Encoder()

    # 使用一张图片进行测试，减少 CPU 内存占用
    test_images = images[:1]

    with torch.no_grad():
        skip1, skip2, skip3, skip4, deepest = full_encoder(test_images)

    print("第一层特征：", skip1.shape)
    print("第二层特征：", skip2.shape)
    print("第三层特征：", skip3.shape)
    print("第四层特征：", skip4.shape)
    print("最深层下采样结果：", deepest.shape)

    # 测试 Bottleneck
    bottleneck = Bottleneck(512, 1024)

    with torch.no_grad():
        bottleneck_output = bottleneck(deepest)

    print("Bottleneck输出：", bottleneck_output.shape)

    # 测试 UpSample
    upsample = UpSample(1024, 512)

    with torch.no_grad():
        upsample_output = upsample(bottleneck_output)

    print("上采样输出：", upsample_output.shape)

    # 测试 Skip Connection
    # 将 Decoder 上采样特征与 Encoder 保存的特征拼接

    skip_connection = torch.cat(
        [upsample_output, skip4],
        dim=1
    )

    print(
        "Skip Connection 拼接结果：",
        skip_connection.shape
    )

    # 测试第一个 DecoderBlock
    decoder1 = DecoderBlock(
        in_channels=1024,
        skip_channels=512,
        out_channels=512
    )

    with torch.no_grad():
        decoder1_output = decoder1(
            bottleneck_output,
            skip4
        )

    print(
        "DecoderBlock输出：",
        decoder1_output.shape
    )

    # 测试完整 Decoder
    decoder = Decoder()

    with torch.no_grad():

        decoder_output = decoder(
            bottleneck_output,
            skip1,
            skip2,
            skip3,
            skip4
        )

    print(
        "完整Decoder输出：",
        decoder_output.shape
    )

    # 测试 OutputLayer
    output_layer = OutputLayer(
        in_channels=64,
        num_classes=1
    )

    with torch.no_grad():
        prediction = output_layer(decoder_output)

    print("最终分割输出：", prediction.shape)

    # ==================================
    # 测试完整 U-Net
    # ==================================

    unet = UNet()

    # 切换到评估模式
    unet.eval()

    # 使用一张图片测试
    with torch.no_grad():

        unet_prediction = unet(test_images)

    print(
        "完整 U-Net 输入：",
        test_images.shape
    )

    print(
        "完整 U-Net 输出：",
        unet_prediction.shape
    )

