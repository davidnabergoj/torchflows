from typing import Tuple, Union

import torch
import torch.nn as nn

from torchflows.bijections.finite.autoregressive.conditioning.transforms import TensorConditionerTransform


class ConvModifier(nn.Module):
    """
    Convolutional layer that transforms an image with size (c, h, w) into an image with size (4, 32, 32).
    """

    def __init__(self,
                 image_shape,
                 c_target: int = 4,
                 h_target: int = 32,
                 w_target: int = 32):
        super().__init__()
        c, h, w = image_shape
        if h >= h_target:
            kernel_height = h - h_target + 1
            padding_height = 0
        else:
            kernel_height = 1 if (h_target - h) % 2 == 0 else 2
            padding_height = ((h_target - h) + kernel_height - 1) // 2
        if w >= w_target:
            kernel_width = w - w_target + 1
            padding_width = 0
        else:
            kernel_width = 1 if (w_target - w) % 2 == 0 else 2
            padding_width = ((w_target - w) + kernel_width - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=c,
            out_channels=c_target,
            kernel_size=(kernel_height, kernel_width),
            padding=(padding_height, padding_width)
        )
        self.output_shape = (c_target, h_target, w_target)

    def forward(self, x):
        return self.conv(x)


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

    def __init__(self,
                 input_shape,
                 n_outputs: int,
                 kernels: Tuple[int, ...] = None):
        """

        :param input_shape: (channels, height, width)
        :param n_outputs:
        """
        super().__init__()

        if kernels is None:
            kernels = (8, 8, 4)
        else:
            assert len(kernels) >= 1

        reducer = ConvModifier(input_shape)

        blocks = [
            self.ConvNetBlock(
                in_channels=reducer.output_shape[0],
                out_channels=kernels[0],
                input_height=reducer.output_shape[1],
                input_width=reducer.output_shape[2],
                use_pooling=min(reducer.output_shape[1], reducer.output_shape[2]) >= 2
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

        self.blocks = nn.ModuleList([reducer] + blocks)

        hidden_size_sqrt: int = 10
        hidden_size = hidden_size_sqrt ** 2
        self.blocks.append(
            ConvModifier(
                image_shape=blocks[-1].output_shape,
                c_target=1,
                h_target=hidden_size_sqrt,
                w_target=hidden_size_sqrt
            )
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=n_outputs
        )

    def forward(self, x):
        batch_shape = x.shape[:-3]
        for block in self.blocks:
            x = block(x)
        x = x.view(*batch_shape, -1)
        x = self.linear(x)
        return x


class ConvNetConditioner(TensorConditionerTransform):
    def __init__(self,
                 input_event_shape: Union[Tuple[int, ...], torch.Size],
                 parameter_shape: Union[Tuple[int, ...], torch.Size],
                 kernels: Tuple[int, ...] = None,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            parameter_shape=parameter_shape,
            output_lower_bound=-2.0,
            output_upper_bound=2.0,
            **kwargs
        )
        self.network = ConvNet(
            input_shape=input_event_shape,
            n_outputs=self.n_transformer_parameters,
            kernels=kernels
        )

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.network(x)
