from typing import Type, Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.finite.autoregressive.conditioning.transforms import ConditionerTransform
from torchflows.bijections.base import Bijection, BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers import ActNorm
from torchflows.bijections.finite.autoregressive.layers_base import CouplingBijection
from torchflows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from torchflows.bijections.finite.multiscale.coupling import make_image_coupling, Checkerboard, \
    ChannelWiseHalfSplit
from torchflows.neural_networks.convnet import ConvNet
from torchflows.neural_networks.resnet import make_resnet18
from torchflows.utils import get_batch_shape


class ConvNetConditioner(ConditionerTransform):
    def __init__(self,
                 input_event_shape: torch.Size,
                 parameter_shape: torch.Size,
                 kernels: Tuple[int, ...] = None,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=None,
            parameter_shape=parameter_shape,
            **kwargs
        )
        self.network = ConvNet(
            input_shape=input_event_shape,
            n_outputs=self.n_transformer_parameters,
            kernels=kernels,
            # output_lower_bound=-8.0,
            # output_upper_bound=8.0,
        )

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.network(x)


class ResNetConditioner(ConditionerTransform):
    def __init__(self,
                 input_event_shape: torch.Size,
                 parameter_shape: torch.Size,
                 **kwargs):
        super().__init__(
            input_event_shape=input_event_shape,
            context_shape=None,
            parameter_shape=parameter_shape,
            **kwargs
        )
        self.network = make_resnet18(
            image_shape=input_event_shape,
            n_outputs=self.n_transformer_parameters
        )

    def predict_theta_flat(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.network(x)


class ConvolutionalCouplingBijection(CouplingBijection):
    def __init__(self,
                 transformer: TensorTransformer,
                 coupling: Union[Checkerboard, ChannelWiseHalfSplit],
                 conditioner='convnet',
                 **kwargs):
        if conditioner == 'convnet':
            conditioner_transform = ConvNetConditioner(
                input_event_shape=coupling.constant_shape,
                parameter_shape=transformer.parameter_shape,
                **kwargs
            )
        elif conditioner == 'resnet':
            conditioner_transform = ResNetConditioner(
                input_event_shape=coupling.constant_shape,
                parameter_shape=transformer.parameter_shape,
                **kwargs
            )
        else:
            raise ValueError(f'Unknown conditioner: {conditioner}')
        super().__init__(transformer, coupling, conditioner_transform, **kwargs)
        self.coupling = coupling

    def get_constant_part(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: tensor with shape (*b, channels, height, width).
        :return: tensor with shape (*b, constant_channels, constant_height, constant_width).
        """
        batch_shape = get_batch_shape(x, self.event_shape)
        return x[..., self.coupling.source_mask].view(*batch_shape, *self.coupling.constant_shape)

    def get_transformed_part(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: tensor with shape (*b, channels, height, width).
        :return: tensor with shape (*b, transformed_channels, transformed_height, constant_width).
        """
        batch_shape = get_batch_shape(x, self.event_shape)
        return x[..., self.coupling.target_mask].view(*batch_shape, *self.coupling.transformed_shape)

    def set_transformed_part(self, x: torch.Tensor, x_transformed: torch.Tensor):
        """

        :param x: tensor with shape (*b, channels, height, width).
        :param x_transformed: tensor with shape (*b, transformed_channels, transformed_height, transformed_width).
        """
        batch_shape = get_batch_shape(x, self.event_shape)
        return x[..., self.coupling.target_mask].view(*batch_shape, *self.coupling.transformed_shape)

    def partition_and_predict_parameters(self, x: torch.Tensor, context: torch.Tensor):
        batch_shape = get_batch_shape(x, self.event_shape)
        super_out = super().partition_and_predict_parameters(x, context)
        return super_out.view(*batch_shape, *self.coupling.transformed_shape,
                              *self.transformer.parameter_shape_per_element)


class CheckerboardCoupling(ConvolutionalCouplingBijection):
    def __init__(self,
                 event_shape,
                 transformer_class: Type[TensorTransformer],
                 alternate: bool = False,
                 **kwargs):
        coupling = make_image_coupling(
            event_shape,
            coupling_type='checkerboard' if not alternate else 'checkerboard_inverted',
        )
        transformer = transformer_class(event_shape=coupling.transformed_shape)
        super().__init__(transformer, coupling, **kwargs)


class NormalizedCheckerboardCoupling(BijectiveComposition):
    def __init__(self, event_shape, **kwargs):
        layers = [
            CheckerboardCoupling(event_shape, **kwargs),
            ActNorm(event_shape)
        ]
        super().__init__(event_shape, layers)


class ChannelWiseCoupling(ConvolutionalCouplingBijection):
    def __init__(self,
                 event_shape,
                 transformer_class: Type[TensorTransformer],
                 alternate: bool = False,
                 **kwargs):
        coupling = make_image_coupling(
            event_shape,
            coupling_type='channel_wise' if not alternate else 'channel_wise_inverted'
        )
        transformer = transformer_class(event_shape=coupling.transformed_shape)
        super().__init__(transformer, coupling, **kwargs)


class NormalizedChannelWiseCoupling(BijectiveComposition):
    def __init__(self, event_shape, **kwargs):
        layers = [
            ChannelWiseCoupling(event_shape, **kwargs),
            ActNorm(event_shape)
        ]
        super().__init__(event_shape, layers)


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
        log_det = torch.zeros(*batch_shape, device=x.device, dtype=x.dtype).to(x)

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
        log_det = torch.zeros(*batch_shape, device=z.device, dtype=z.dtype).to(z)

        four_channels, half_height, half_width = z.shape[-3:]
        assert four_channels % 4 == 0
        width = 2 * half_width
        height = 2 * half_height
        channels = four_channels // 4

        out = torch.empty(size=(*batch_shape, channels, height, width), device=z.device, dtype=z.dtype).to(z)
        out[..., ::2, ::2] = z[..., 0:channels, :, :]
        out[..., ::2, 1::2] = z[..., channels:2 * channels, :, :]
        out[..., 1::2, ::2] = z[..., 2 * channels:3 * channels, :, :]
        out[..., 1::2, 1::2] = z[..., 3 * channels:4 * channels, :, :]
        return out, log_det


class MultiscaleBijectiveComposition(Bijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 transformer_class: Type[TensorTransformer],
                 n_blocks: int,
                 n_checkerboard_layers: int = 3,
                 n_channel_wise_layers: int = 3,
                 use_resnet: bool = False,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        if n_blocks < 1:
            raise ValueError

        self.n_blocks = n_blocks
        self.checkerboard_layers = nn.ModuleList([
            NormalizedCheckerboardCoupling(
                event_shape,
                transformer_class=transformer_class,
                alternate=i % 2 == 1,
                conditioner='resnet' if use_resnet else 'convnet'
            )
            for i in range(n_checkerboard_layers + (0 if n_blocks > 1 else 1))
        ])

        if self.n_blocks > 1:
            self.squeeze = Squeeze(event_shape)
            self.channel_wise_layers = nn.ModuleList([
                NormalizedChannelWiseCoupling(
                    self.squeeze.transformed_event_shape,
                    transformer_class=transformer_class,
                    alternate=i % 2 == 1,
                    conditioner='resnet' if use_resnet else 'convnet'
                )
                for i in range(n_channel_wise_layers)
            ])

            self.alt_squeeze = Squeeze(event_shape, alternate=True)

            small_event_shape = (
                self.alt_squeeze.transformed_event_shape[0] // 2,
                *self.alt_squeeze.transformed_event_shape[1:]
            )
            self.small_bijection = MultiscaleBijectiveComposition(
                event_shape=small_event_shape,
                transformer_class=transformer_class,
                n_blocks=self.n_blocks - 1,
                **kwargs
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(size=get_batch_shape(x, event_shape=self.event_shape)).to(x)

        # Propagate through checkerboard layers
        for layer in self.checkerboard_layers:
            x, log_det_layer = layer.forward(x, context=context)
            log_det += log_det_layer

        if self.n_blocks > 1:
            # Propagate through channel-wise layers
            x, _ = self.squeeze.forward(x, context=context)
            for layer in self.channel_wise_layers:
                x, log_det_layer = layer.forward(x, context=context)
                log_det += log_det_layer
            x, _ = self.squeeze.inverse(x, context=context)

            # Chunk and apply small bijection
            x, _ = self.alt_squeeze.forward(x, context=context)
            x_const, x_rest = torch.chunk(x, 2, dim=-3)  # channel dimension split (..., c, h, w)
            x_rest, log_det_layer = self.small_bijection.forward(x_rest, context=context)
            log_det += log_det_layer
            x = torch.cat((x_const, x_rest), dim=-3)  # channel dimension concatenation
            x, _ = self.alt_squeeze.inverse(x, context=context)

        z = x
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(size=get_batch_shape(z, event_shape=self.event_shape)).to(z)

        if self.n_blocks > 1:
            # Chunk and apply small bijection
            z, _ = self.alt_squeeze.forward(z, context=context)
            z_const, z_rest = torch.chunk(z, 2, dim=-3)  # channel dimension split (..., c, h, w)
            z_rest, log_det_layer = self.small_bijection.inverse(z_rest, context=context)
            log_det += log_det_layer
            z = torch.cat((z_const, z_rest), dim=-3)  # channel dimension concatenation
            z, _ = self.alt_squeeze.inverse(z, context=context)

            # Propagate through channel-wise layers
            z, _ = self.squeeze.forward(z, context=context)
            for layer in self.channel_wise_layers[::-1]:
                z, log_det_layer = layer.inverse(z, context=context)
                log_det += log_det_layer
            z, _ = self.squeeze.inverse(z, context=context)

        # Propagate through checkerboard layers
        for layer in self.checkerboard_layers[::-1]:
            z, log_det_layer = layer.inverse(z, context=context)
            log_det += log_det_layer

        x = z
        return x, log_det
