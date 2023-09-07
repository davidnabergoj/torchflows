import torch

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import MADE, FeedForward
from normalizing_flows.bijections.finite.autoregressive.conditioners.coupling import Coupling
from normalizing_flows.bijections.finite.autoregressive.conditioners.masked import MaskedAutoregressive
from normalizing_flows.bijections.finite.autoregressive.layers_base import AutoregressiveLayer, \
    ForwardMaskedAutoregressiveLayer, InverseMaskedAutoregressiveLayer, ElementwiseLayer, CouplingLayer
from normalizing_flows.bijections.finite.autoregressive.transformers.affine import Scale, Affine, Shift
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear_rational import LinearRational
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Inverse
from normalizing_flows.bijections.finite.autoregressive.transformers.combination import (
    DeepSigmoidNetwork,
    UnconstrainedMonotonicNeuralNetwork
)


class ElementwiseAffine(ElementwiseLayer):
    def __init__(self, event_shape, **kwargs):
        super().__init__(Affine(event_shape, **kwargs), n_transformer_parameters=2)


class ElementwiseScale(ElementwiseLayer):
    def __init__(self, event_shape, **kwargs):
        super().__init__(Scale(event_shape, **kwargs), n_transformer_parameters=1)


class ElementwiseShift(ElementwiseLayer):
    def __init__(self, event_shape):
        super().__init__(Shift(event_shape), n_transformer_parameters=1)


class ElementwiseRQSpline(ElementwiseLayer):
    def __init__(self, event_shape, **kwargs):
        transformer = RationalQuadratic(event_shape, **kwargs)
        super().__init__(transformer, n_transformer_parameters=transformer.n_bins * 3 - 1)

        # Initialize spline parameters to define a linear transform
        # TODO remove this
        with torch.no_grad():
            self.conditioner_transform.theta[..., :2 * transformer.n_bins] = 0.0
            self.conditioner_transform.theta[..., 2 * transformer.n_bins:] = transformer.boundary_u_delta


class AffineCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        transformer = Affine(event_shape=event_shape)
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class InverseAffineCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        transformer = Inverse(Affine(event_shape=event_shape))
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class ShiftCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer = Shift(event_shape=event_shape)
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class LRSCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        assert n_bins >= 1
        transformer = LinearRational(event_shape=event_shape, n_bins=n_bins)
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class RQSCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class DSCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_sigmoid_layers: int = 2,
                 **kwargs):
        transformer = DeepSigmoidNetwork(event_shape=event_shape, n_layers=n_sigmoid_layers)
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class InverseDSCoupling(CouplingLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_sigmoid_layers: int = 2,
                 **kwargs):
        transformer = Inverse(DeepSigmoidNetwork(event_shape=event_shape, n_layers=n_sigmoid_layers))
        conditioner = Coupling(constants=transformer.default_parameters, event_shape=event_shape)
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_shape=conditioner.input_shape,
            output_shape=conditioner.output_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(conditioner, transformer, conditioner_transform)


class LinearAffineCoupling(AffineCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearRQSCoupling(RQSCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearLRSCoupling(LRSCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearShiftCoupling(ShiftCoupling):
    def __init__(self, event_shape: torch.Size, **kwargs):
        super().__init__(event_shape, **kwargs, n_layers=1)


class AffineForwardMaskedAutoregressive(ForwardMaskedAutoregressiveLayer):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer = Affine(event_shape=event_shape)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=transformer.n_parameters,
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
        transformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=transformer.n_parameters,
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
                 **kwargs):
        transformer = Inverse(Affine(event_shape=event_shape))
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=transformer.n_parameters,
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
        transformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=transformer.n_parameters,
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
        conditioner_transform = MADE(
            input_shape=event_shape,
            output_shape=event_shape,
            n_output_parameters=transformer.n_parameters,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )
