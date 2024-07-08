from typing import Type, Union, Tuple

import torch

from normalizing_flows.bijections import BijectiveComposition
from normalizing_flows.bijections.finite.autoregressive.conditioning.transforms import ConditionerTransform
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.finite.autoregressive.layers_base import CouplingBijection
from normalizing_flows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from normalizing_flows.bijections.finite.multiscale.coupling import make_image_coupling, Checkerboard, \
    ChannelWiseHalfSplit
from normalizing_flows.neural_networks.convnet import ConvNet
from normalizing_flows.neural_networks.resnet import make_resnet18
from normalizing_flows.utils import get_batch_shape


class FactoredBijection(Bijection):
    """
    Factored bijection class.

    Partitions the input tensor x into parts x_A and x_B, then applies a bijection to x_A independently of x_B while
    keeping x_B identical.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 small_bijection: Bijection,
                 small_bijection_mask: torch.Tensor,
                 **kwargs):
        """

        :param event_shape: shape of input event x.
        :param small_bijection: bijection applied to transformed event x_A.
        :param small_bijection_mask: boolean mask that selects which elements of event x correspond to the transformed
            event x_A.
        :param kwargs:
        """
        super().__init__(event_shape, **kwargs)

        # Check that shapes are correct
        event_size = torch.prod(torch.as_tensor(event_shape))
        transformed_event_size = torch.prod(torch.as_tensor(small_bijection.event_shape))
        assert event_size >= transformed_event_size

        assert small_bijection_mask.shape == event_shape

        self.transformed_event_mask = small_bijection_mask
        self.small_bijection = small_bijection

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        transformed, log_det = self.small_bijection.forward(
            x[..., self.transformed_event_mask].view(*batch_shape, *self.small_bijection.event_shape),
            context
        )
        out = x.clone()
        out[..., self.transformed_event_mask] = transformed.view(*batch_shape, -1)
        return out, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        transformed, log_det = self.small_bijection.inverse(
            z[..., self.transformed_event_mask].view(*batch_shape, *self.small_bijection.transformed_shape),
            context
        )
        out = z.clone()
        out[..., self.transformed_event_mask] = transformed.view(*batch_shape, -1)
        return out, log_det


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
            kernels=kernels
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
            coupling_type='checkerboard' if not alternate else 'checkerboard_inverted'
        )
        transformer = transformer_class(event_shape=coupling.transformed_shape)
        super().__init__(transformer, coupling, **kwargs)


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
                 n_checkerboard_layers: int = 2,
                 n_channel_wise_layers: int = 2,
                 use_squeeze_layer: bool = True,
                 use_resnet: bool = False,
                 **kwargs):
        checkerboard_layers = [
            CheckerboardCoupling(
                input_event_shape,
                transformer_class,
                alternate=i % 2 == 1,
                conditioner='resnet' if use_resnet else 'convnet'
            )
            for i in range(n_checkerboard_layers)
        ]
        squeeze_layer = Squeeze(input_event_shape)
        channel_wise_layers = [
            ChannelWiseCoupling(
                squeeze_layer.transformed_event_shape,
                transformer_class,
                alternate=i % 2 == 1,
                conditioner='resnet' if use_resnet else 'convnet'
            )
            for i in range(n_channel_wise_layers)
        ]
        if use_squeeze_layer:
            layers = [*checkerboard_layers, squeeze_layer, *channel_wise_layers]
        else:
            layers = [*checkerboard_layers, *channel_wise_layers]
        super().__init__(input_event_shape, layers, **kwargs)
        self.transformed_shape = squeeze_layer.transformed_event_shape if use_squeeze_layer else input_event_shape
