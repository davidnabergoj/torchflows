from typing import Type, Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.bijections import BijectiveComposition, CouplingBijection
from normalizing_flows.bijections.finite.autoregressive.conditioning.transforms import FeedForward, ResidualFeedForward
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from normalizing_flows.bijections.finite.multiscale.coupling import make_image_coupling
from normalizing_flows.utils import get_batch_shape


class CheckerboardCoupling(CouplingBijection):
    def __init__(self,
                 event_shape,
                 transformer_class: Type[TensorTransformer],
                 alternate: bool = False,
                 **kwargs):
        coupling = make_image_coupling(
            event_shape,
            coupling_type='checkerboard' if not alternate else 'checkerboard_inverted'
        )
        transformer = transformer_class(event_shape=torch.Size((coupling.target_event_size,)))
        conditioner_transform = ResidualFeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            nonlinearity=nn.Tanh,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform, **kwargs)


class ChannelWiseCoupling(CouplingBijection):
    def __init__(self,
                 event_shape,
                 transformer_class: Type[TensorTransformer],
                 alternate: bool = False,
                 **kwargs):
        coupling = make_image_coupling(
            event_shape,
            coupling_type='channel_wise' if not alternate else 'channel_wise_inverted'
        )
        transformer = transformer_class(event_shape=torch.Size((coupling.target_event_size,)))
        conditioner_transform = ResidualFeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            nonlinearity=nn.Tanh,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform, **kwargs)


class Squeeze(Bijection):
    """
    Squeeze a batch of tensors with shape (*batch_shape, channels, height, width) into shape
        (*batch_shape, 4 * channels, height / 2, width / 2).
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        # Check shape length
        if len(event_shape) != 3:
            raise ValueError(f"Event shape must have three components, but got {len(event_shape)}")
        # Check that height and width are divisible by two
        if event_shape[1] % 2 != 0:
            raise ValueError(f"Event dimension 1 must be divisible by 2, but got {event_shape[1]}")
        if event_shape[2] % 2 != 0:
            raise ValueError(f"Event dimension 2 must be divisible by 2, but got {event_shape[2]}")
        super().__init__(event_shape, **kwargs)
        c, h, w = event_shape
        self.transformed_event_shape = torch.Size((4 * c, h // 2, w // 2))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Squeeze tensor with shape (*batch_shape, channels, height, width) into tensor with shape
            (*batch_shape, 4 * channels, height // 2, width // 2).
        """
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(*batch_shape, device=x.device, dtype=x.dtype)

        channels, height, width = x.shape[-3:]
        assert height % 2 == 0
        assert width % 2 == 0

        out = torch.concatenate([
            x[..., ::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, ::2],
            x[..., 1::2, 1::2]
        ], dim=-3)
        return out, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Squeeze tensor with shape (*batch_shape, 4 * channels, height // 2, width // 2) into tensor with shape
            (*batch_shape, channels, height, width).
        """
        batch_shape = get_batch_shape(z, self.transformed_event_shape)
        log_det = torch.zeros(*batch_shape, device=z.device, dtype=z.dtype)

        four_channels, half_height, half_width = z.shape[-3:]
        assert four_channels % 4 == 0
        width = 2 * half_width
        height = 2 * half_height
        channels = four_channels // 4

        out = torch.empty(size=(*batch_shape, channels, height, width), device=z.device, dtype=z.dtype)
        out[..., ::2, ::2] = z[..., 0:channels, :, :]
        out[..., ::2, 1::2] = z[..., channels:2 * channels, :, :]
        out[..., 1::2, ::2] = z[..., 2 * channels:3 * channels, :, :]
        out[..., 1::2, 1::2] = z[..., 3 * channels:4 * channels, :, :]
        return out, log_det


class MultiscaleBijection(BijectiveComposition):
    def __init__(self,
                 input_event_shape,
                 transformer_class: Type[TensorTransformer],
                 n_checkerboard_layers: int = 3,
                 n_channel_wise_layers: int = 3,
                 use_squeeze_layer: bool = True,
                 **kwargs):
        channels, height, width = input_event_shape[-3:]
        resolution = min(width, height) // 2
        checkerboard_layers = [
            CheckerboardCoupling(
                input_event_shape,
                transformer_class,
                alternate=i % 2 == 1,
                resolution=resolution
            )
            for i in range(n_checkerboard_layers)
        ]
        squeeze_layer = Squeeze(input_event_shape)
        channel_wise_layers = [
            ChannelWiseCoupling(
                squeeze_layer.transformed_event_shape,
                transformer_class,
                alternate=i % 2 == 1,
                resolution=resolution
            )
            for i in range(n_channel_wise_layers)
        ]
        if use_squeeze_layer:
            layers = [*checkerboard_layers, squeeze_layer, *channel_wise_layers]
        else:
            layers = [*checkerboard_layers, *channel_wise_layers]
        super().__init__(input_event_shape, layers, **kwargs)
        self.transformed_shape = squeeze_layer.transformed_event_shape if use_squeeze_layer else input_event_shape
