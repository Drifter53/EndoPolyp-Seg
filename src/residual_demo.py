import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # 主分支：第一次卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        # 主分支：第二次卷积
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # 判断是否需要维度匹配
        if in_channels != out_channels or stride != 1:

            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        # Shortcut 分支
        identity = self.shortcut(x)

        # 主分支
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        # 检查形状
        print("主分支形状：", out.shape)
        print("Shortcut形状：", identity.shape)

        # 残差相加
        out = out + identity

        out = self.relu(out)

        return out


if __name__ == "__main__":

    # 模拟输入
    x = torch.randn(1, 64, 64, 64)

    # 创建残差模块
    model = ResidualBlock(
        in_channels=64,
        out_channels=128,
        stride=2
    )

    with torch.no_grad():

        output = model(x)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)