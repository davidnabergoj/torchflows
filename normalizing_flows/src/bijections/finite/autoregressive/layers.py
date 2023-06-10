import math

import torch

from normalizing_flows.src.bijections.finite.autoregressive.conditioner_transforms import MADE, FeedForward
from normalizing_flows.src.bijections.finite.autoregressive.conditioners import Coupling, MaskedAutoregressive
from normalizing_flows.src.bijections.finite.autoregressive.layers_base import AutoregressiveLayer, \
    ForwardMaskedAutoregressiveLayer, InverseMaskedAutoregressiveLayer
from normalizing_flows.src.bijections.finite.autoregressive.transformers import Affine, Shift, InverseAffine, RationalQuadraticSpline


class AffineCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 scale_transform: callable = torch.exp,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        default_log_scale = 0.0
        default_shift = 0.0
        conditioner = Coupling(constants=torch.tensor([default_log_scale, default_shift]), event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=2,
            context_shape=context_shape,
            **kwargs
        )
        transformer = Affine(event_shape=event_shape, scale_transform=scale_transform)
        super().__init__(conditioner, transformer, conditioner_transform)


class InverseAffineCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 scale_transform: callable = torch.exp,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        default_log_scale = 0.0
        default_shift = 0.0
        conditioner = Coupling(constants=torch.tensor([default_log_scale, default_shift]), event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=2,
            context_shape=context_shape,
            **kwargs
        )
        transformer = InverseAffine(event_shape=event_shape, scale_transform=scale_transform)
        super().__init__(conditioner, transformer, conditioner_transform)


class ShiftCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        default_shift = 0.0
        conditioner = Coupling(constants=torch.tensor([default_shift]), event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=1,
            context_shape=context_shape,
            **kwargs
        )
        transformer = Shift(event_shape=event_shape)
        super().__init__(conditioner, transformer, conditioner_transform)


class RQSCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 boundary: float = 1.0,
                 **kwargs):
        assert n_bins >= 1
        default_unconstrained_widths = torch.zeros(n_bins)
        default_unconstrained_heights = torch.zeros(n_bins)
        default_unconstrained_derivatives = torch.full(size=(n_bins - 1,), fill_value=math.log(math.expm1(1)))
        constants = torch.cat([
            default_unconstrained_widths,
            default_unconstrained_heights,
            default_unconstrained_derivatives
        ])

        conditioner = Coupling(constants=constants, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=3 * n_bins - 1,
            context_shape=context_shape,
            **kwargs
        )
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins, boundary=boundary)
        super().__init__(conditioner, transformer, conditioner_transform)


class LinearAffineCoupling(AffineCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearRQSCoupling(RQSCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearShiftCoupling(ShiftCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class AffineForwardMaskedAutoregressive(ForwardMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 scale_transform: callable = torch.exp,
                 **kwargs):
        transformer = Affine(event_shape=event_shape, scale_transform=scale_transform)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=2,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class RQSForwardMaskedAutoregressive(ForwardMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 boundary: float = 1.0,
                 **kwargs):
        assert n_bins >= 1
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins, boundary=boundary)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=3 * n_bins - 1,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 scale_transform: callable = torch.exp,
                 **kwargs):
        transformer = InverseAffine(event_shape=event_shape, scale_transform=scale_transform)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=2,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class RQSInverseMaskedAutoregressive(InverseMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 boundary: float = 1.0,
                 **kwargs):
        assert n_bins >= 1
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins, boundary=boundary)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=3 * n_bins - 1,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )
