import torch

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import MADE, FeedForward
from normalizing_flows.bijections.finite.autoregressive.conditioners.coupling_masks import HalfSplit
from normalizing_flows.bijections.finite.autoregressive.conditioners.masked import MaskedAutoregressive
from normalizing_flows.bijections.finite.autoregressive.layers_base import ForwardMaskedAutoregressiveBijection, \
    InverseMaskedAutoregressiveBijection, ElementwiseBijection, CouplingBijection
from normalizing_flows.bijections.finite.autoregressive.transformers.affine import Scale, Affine, Shift
from normalizing_flows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from normalizing_flows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network import \
    UnconstrainedMonotonicNeuralNetwork
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear_rational import LinearRational
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid import (
    DeepSigmoid
)
from normalizing_flows.bijections.base import invert


# TODO move elementwise bijections, coupling bijections, and masked autoregressive bijections into separate files.
class ElementwiseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        transformer = Affine(event_shape, **kwargs)
        super().__init__(transformer)


class ElementwiseScale(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        transformer = Scale(event_shape, **kwargs)
        super().__init__(transformer)


class ElementwiseShift(ElementwiseBijection):
    def __init__(self, event_shape):
        transformer = Shift(event_shape)
        super().__init__(transformer)


class ElementwiseRQSpline(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        transformer = RationalQuadratic(event_shape, **kwargs)
        super().__init__(transformer)


class AffineCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        coupling_mask = HalfSplit(event_shape)
        transformer = Affine(event_shape=torch.Size((coupling_mask.transformed_event_size,)))
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


class InverseAffineCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        coupling_mask = HalfSplit(event_shape)
        transformer = Affine(event_shape=torch.Size((coupling_mask.transformed_event_size,))).invert()
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


class ShiftCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        coupling_mask = HalfSplit(event_shape)
        transformer = Shift(event_shape=torch.Size((coupling_mask.transformed_event_size,)))
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


class LRSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        assert n_bins >= 1
        coupling_mask = HalfSplit(event_shape)
        transformer = LinearRational(event_shape=torch.Size((coupling_mask.transformed_event_size,)), n_bins=n_bins)
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


class RQSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        coupling_mask = HalfSplit(event_shape)
        transformer = RationalQuadratic(event_shape=torch.Size((coupling_mask.transformed_event_size,)), n_bins=n_bins)
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


class DSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 **kwargs):
        coupling_mask = HalfSplit(event_shape)
        transformer = DeepSigmoid(
            event_shape=torch.Size((coupling_mask.transformed_event_size,)),
            n_hidden_layers=n_hidden_layers
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling_mask.constant_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling_mask, conditioner_transform)


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


class AffineForwardMaskedAutoregressive(ForwardMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer: ScalarTransformer = Affine(event_shape=event_shape)
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class RQSForwardMaskedAutoregressive(ForwardMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class LRSForwardMaskedAutoregressive(ForwardMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer: ScalarTransformer = LinearRational(event_shape=event_shape, n_bins=n_bins)
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer: ScalarTransformer = invert(Affine(event_shape=event_shape))
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class RQSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        assert n_bins >= 1
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )


class UMNNMaskedAutoregressive(ForwardMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 1,
                 hidden_dim: int = 5,
                 **kwargs):
        transformer: ScalarTransformer = UnconstrainedMonotonicNeuralNetwork(
            event_shape=event_shape,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim
        )
        conditioner_transform = MADE(
            input_event_shape=event_shape,
            output_event_shape=event_shape,
            parameter_shape=transformer.parameter_shape_per_element,
            context_shape=context_shape,
            **kwargs
        )
        conditioner = MaskedAutoregressive()
        super().__init__(
            conditioner=conditioner,
            transformer=transformer,
            conditioner_transform=conditioner_transform
        )
