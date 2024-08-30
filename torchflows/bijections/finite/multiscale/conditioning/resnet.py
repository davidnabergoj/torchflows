from typing import Tuple

from torchflows.bijections.finite.autoregressive.conditioning.transforms import TensorConditionerTransform
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchflows.bijections.finite.multiscale.conditioning.classic import ConvModifier


class BasicResidualBlock(nn.Module):
    """
    Basic residual block. Keeps image height and width the same.
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def h(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(y)))
        return z

    def forward(self, x):
        return x + self.h(x)


class BasicResidualBlockGroup(nn.Module):
    def __init__(self, in_channels: int, n_blocks: int, hidden_channels: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([BasicResidualBlock(in_channels, hidden_channels) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Doubles the number of channels, halves height and width.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv2d(x)


class ResNet(nn.Module):
    """
    ResNet class.
    """

    def __init__(self,
                 c,
                 h,
                 w,
                 hidden_size: int = 100,
                 n_outputs: int = 10,
                 n_blocks: Tuple[int, int, int] = (1, 1, 1)):
        """

        :param c: number of input image channels.
        :param h: input image height.
        :param w: input image width.
        :param hidden_size: number of hidden units at the last linear layer.
        :param n_outputs: number of outputs.
        """
        super(ResNet, self).__init__()
        self.modifier = ConvModifier((c, h, w), c_target=4, h_target=32, w_target=32)  # to `(4, 32, 32)`

        self.stage1 = BasicResidualBlockGroup(4, n_blocks=n_blocks[0])  # (4, 32, 32)
        self.down1 = BottleneckBlock(4)  # (8, 16, 16)

        self.stage2 = BasicResidualBlockGroup(8, n_blocks=n_blocks[1])  # (8, 16, 16)
        self.down2 = BottleneckBlock(8)  # (16, 8, 8)

        self.stage3 = BasicResidualBlockGroup(16, n_blocks=n_blocks[2])  # (16, 8, 8)
        self.down3 = BottleneckBlock(16)  # (32, 4, 4), note: 32 * 4 * 4 = 512 (for linear layer)

        self.linear1 = nn.Linear(512, hidden_size)  # 32 * 4 * 4 = 512
        self.linear2 = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        """
        :param x: tensor with shape (*b, channels, height, width).
        :return:
        """
        batch_shape = x.shape[:-3]

        out = self.modifier(x)

        out = self.stage1(out)
        out = self.down1(out)

        out = self.stage2(out)
        out = self.down2(out)

        out = self.stage3(out)
        out = self.down3(out)

        out = self.linear1(out.view(*batch_shape, 512))
        out = F.leaky_relu(out)
        out = self.linear2(out)
        return out


class ResNetConditioner(TensorConditionerTransform):
    def __init__(self,
                 input_event_shape: torch.Size,
                 parameter_shape: torch.Size,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=None,
            parameter_shape=parameter_shape,
            output_lower_bound=-2.0,
            output_upper_bound=2.0,
            **kwargs
        )
        self.network = ResNet(
            *input_event_shape,
            n_outputs=self.n_transformer_parameters
        )

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.network(x)

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn((15, 3, 77, 13))
    rn = ResNet(3, 77, 13, n_outputs=7)
    y = rn(x)
    print(y.shape)