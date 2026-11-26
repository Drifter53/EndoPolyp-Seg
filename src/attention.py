
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):

    def __init__(
        self,
        F_g,
        F_l,
        F_int
    ):
        super().__init__()

        # Decoder 特征的 1×1 卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(F_int)
        )

        # Encoder 特征的 1×1 卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(F_int)
        )

        # 生成注意力权重
        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):

        # 1. 分别进行特征映射
        x1 = self.W_x(x)
        g1 = self.W_g(g)

        # 2. 如果空间尺寸不同，调整 g1
        if g1.shape[2:] != x1.shape[2:]:

            g1 = F.interpolate(
                g1,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # 3. 特征相加并经过 ReLU
        f = self.relu(x1 + g1)

        # 4. 计算注意力权重
        alpha = self.psi(f)

        # 5. 对 Encoder 特征进行加权
        output = x * alpha

        return output

if __name__ == "__main__":

    # 模拟 Encoder 特征
    x = torch.randn(4, 256, 32, 32)

    # 模拟 Decoder 特征
    g = torch.randn(4, 512, 32, 32)

    # 创建 Attention Gate
    attention = AttentionGate(
        F_g=512,
        F_l=256,
        F_int=128
    )

    # 前向传播
    output = attention(x, g)

    print("Encoder 输入：", x.shape)
    print("Decoder 输入：", g.shape)
    print("Attention 输出：", output.shape)
