from typing import Tuple

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    class ConvNetBlock(nn.Module):
        def __init__(self, in_channels, out_channels, input_height, input_width, use_pooling: bool = True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d(2) if use_pooling else nn.Identity()

            if use_pooling:
                self.output_shape = (out_channels, input_height // 2, input_width // 2)
            else:
                self.output_shape = (out_channels, input_height, input_width)

        def forward(self, x):
            return self.bn(self.pool(torch.relu(self.conv(x))))

    def __init__(self, input_shape, n_outputs: int, kernels: Tuple[int, ...] = None):
        """

        :param input_shape: (channels, height, width)
        :param n_outputs:
        """
        super().__init__()
        channels, height, width = input_shape

        if kernels is None:
            kernels = (64, 64, 32, 4)
        else:
            assert len(kernels) >= 1

        blocks = [
            self.ConvNetBlock(
                in_channels=channels,
                out_channels=kernels[0],
                input_height=height,
                input_width=width,
                use_pooling=min(height, width) >= 2
            )
        ]
        for i in range(len(kernels) - 1):
            blocks.append(
                self.ConvNetBlock(
                    in_channels=kernels[i],
                    out_channels=kernels[i + 1],
                    input_height=blocks[i].output_shape[1],
                    input_width=blocks[i].output_shape[2],
                    use_pooling=min(blocks[i].output_shape[1], blocks[i].output_shape[2]) >= 2
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(
            in_features=int(torch.prod(torch.as_tensor(self.blocks[-1].output_shape))),
            out_features=n_outputs
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    image_shape = (1, 36, 29)
    images = torch.randn(size=(11, *image_shape))
    net = ConvNet(input_shape=image_shape, n_outputs=77)
    out = net(images)
    print(f'{out.shape = }')
