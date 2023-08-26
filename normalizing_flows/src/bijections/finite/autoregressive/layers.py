import math

import torch

from normalizing_flows.src.bijections.finite.autoregressive.conditioner_transforms import MADE, FeedForward
from normalizing_flows.src.bijections.finite.autoregressive.conditioners import Coupling, MaskedAutoregressive
from normalizing_flows.src.bijections.finite.autoregressive.layers_base import AutoregressiveLayer, \
    ForwardMaskedAutoregressiveLayer, InverseMaskedAutoregressiveLayer, ElementwiseLayer
from normalizing_flows.src.bijections.finite.autoregressive.transformers import Affine, Shift, InverseAffine, \
    RationalQuadraticSpline
from normalizing_flows.src.bijections.finite.autoregressive.transformers.combination import SigmoidTransform, \
    DeepSigmoidNetwork, InverseDeepSigmoidNetwork, UnconstrainedMonotonicNeuralNetwork


class ElementwiseAffine(ElementwiseLayer):
    def __init__(self, event_shape, **kwargs):
        super().__init__(Affine(event_shape, **kwargs), n_transformer_parameters=2)


class ElementwiseShift(ElementwiseLayer):
    def __init__(self, event_shape):
        super().__init__(Shift(event_shape), n_transformer_parameters=1)


class ElementwiseRQSpline(ElementwiseLayer):
    def __init__(self, event_shape, **kwargs):
        transformer = RationalQuadraticSpline(event_shape, **kwargs)
        super().__init__(transformer, n_transformer_parameters=transformer.n_bins * 3 - 1)

        # Initialize spline parameters to define a linear transform
        with torch.no_grad():
            self.conditioner_transform.theta[..., :2 * transformer.n_bins] = 0.0
            self.conditioner_transform.theta[..., 2 * transformer.n_bins:] = transformer.boundary_u_delta


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
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins)
        super().__init__(conditioner, transformer, conditioner_transform)


class DSCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_sigmoid_layers: int = 2,
                 hidden_dim: int = 8,
                 **kwargs):
        default_unconstrained_scale = torch.full(size=(hidden_dim,), fill_value=math.log(math.expm1(1.0)))
        default_shift = torch.zeros(hidden_dim)
        default_unconstrained_convex_weights = torch.ones(hidden_dim)
        single_component_constants = torch.cat([
            default_unconstrained_scale,
            default_shift,
            default_unconstrained_convex_weights
        ])
        conditioner = Coupling(
            constants=torch.cat([single_component_constants for _ in range(n_sigmoid_layers)]),
            event_shape=event_shape
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=3 * hidden_dim * n_sigmoid_layers,
            context_shape=context_shape,
            **kwargs
        )
        transformer = DeepSigmoidNetwork(event_shape=event_shape, n_layers=n_sigmoid_layers)
        super().__init__(conditioner, transformer, conditioner_transform)


class InverseDSCoupling(AutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_sigmoid_layers: int = 2,
                 hidden_dim: int = 8,
                 **kwargs):
        default_unconstrained_scale = torch.full(size=(hidden_dim,), fill_value=math.log(math.expm1(1.0)))
        default_shift = torch.zeros(hidden_dim)
        default_unconstrained_convex_weights = torch.ones(hidden_dim)
        single_component_constants = torch.cat([
            default_unconstrained_scale,
            default_shift,
            default_unconstrained_convex_weights
        ])
        conditioner = Coupling(
            constants=torch.cat([single_component_constants for _ in range(n_sigmoid_layers)]),
            event_shape=event_shape
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=3 * hidden_dim * n_sigmoid_layers,
            context_shape=context_shape,
            **kwargs
        )
        transformer = InverseDeepSigmoidNetwork(event_shape=event_shape, n_layers=n_sigmoid_layers)
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
                 **kwargs):
        assert n_bins >= 1
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins)
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
                 **kwargs):
        assert n_bins >= 1
        transformer = RationalQuadraticSpline(event_shape=event_shape, n_bins=n_bins)
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


class UMNNForwardMaskedAutoregressive(ForwardMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 1,
                 hidden_dim: int = 5,
                 **kwargs):
        transformer = UnconstrainedMonotonicNeuralNetwork(
            event_shape=event_shape,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim
        )
        assert n_hidden_layers >= 1
        n_output_parameters = (
                2 * hidden_dim  # Input to h1 (W, b)
                + n_hidden_layers * (hidden_dim ** 2 + hidden_dim)  # h1 to h2 (W, b) for all hidden layers
                + hidden_dim + 1  # hn to output (W, b)
        )

        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=n_output_parameters,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )
