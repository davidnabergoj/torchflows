from typing import Tuple, List

import torch

from torchflows.bijections.finite.autoregressive.conditioning.transforms import FeedForward
from torchflows.bijections.finite.autoregressive.conditioning.coupling_masks import make_coupling
from torchflows.bijections.finite.autoregressive.layers_base import MaskedAutoregressiveBijection, \
    InverseMaskedAutoregressiveBijection, ElementwiseBijection, CouplingBijection
from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Scale, Affine, Shift, InverseAffine
from torchflows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from torchflows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network import \
    UnconstrainedMonotonicNeuralNetwork
from torchflows.bijections.finite.autoregressive.transformers.spline.linear_rational import LinearRational
from torchflows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid import (
    DeepSigmoid,
    DenseSigmoid,
    DeepDenseSigmoid
)
from torchflows.bijections.base import invert


class ElementwiseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        transformer = Affine(event_shape, **kwargs)
        super().__init__(transformer)


class ElementwiseInverseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        transformer = InverseAffine(event_shape, **kwargs)
        super().__init__(transformer)


class ActNorm(ElementwiseInverseAffine):
    def __init__(self, event_shape, **kwargs):
        super().__init__(event_shape, **kwargs)
        self.first_training_batch_pass: bool = True
        self.value.requires_grad_(False)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: x.shape = (*batch_shape, *event_shape)
        :param context:
        :return:
        """
        if self.training and self.first_training_batch_pass:
            batch_shape = x.shape[:-len(self.event_shape)]
            n_batch_dims = len(batch_shape)
            self.first_training_batch_pass = False
            shift = torch.mean(x, dim=list(range(n_batch_dims)))[..., None].to(self.value)
            if torch.prod(torch.as_tensor(batch_shape)) == 1:
                scale = torch.ones_like(shift)  # unit scale if unable to estimate
            else:
                scale = torch.std(x, dim=list(range(n_batch_dims)))[..., None].to(self.value)
            unconstrained_scale = self.transformer.unconstrain_scale(scale)
            self.value.data = torch.concatenate([unconstrained_scale, shift], dim=-1).data
        return super().forward(x, context)


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
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = Affine(event_shape=torch.Size((coupling.target_event_size,)))
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class InverseAffineCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if event_shape == (1,):
            raise ValueError
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = invert(Affine(event_shape=torch.Size((coupling.target_event_size,))))
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class ShiftCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = Shift(event_shape=torch.Size((coupling.target_event_size,)))
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class LRSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        assert n_bins >= 1
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = LinearRational(event_shape=torch.Size((coupling.target_event_size,)), n_bins=n_bins)
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class RQSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = RationalQuadratic(event_shape=torch.Size((coupling.target_event_size,)), n_bins=n_bins)
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class DeepSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = DeepSigmoid(
            event_shape=torch.Size((coupling.target_event_size,)),
            n_hidden_layers=n_hidden_layers
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class DeepSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DeepSigmoid(
            event_shape=torch.Size(event_shape),
            n_hidden_layers=n_hidden_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DeepSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DeepSigmoid(
            event_shape=torch.Size(event_shape),
            n_hidden_layers=n_hidden_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DenseSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_dense_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = DenseSigmoid(
            event_shape=torch.Size((coupling.target_event_size,)),
            n_dense_layers=n_dense_layers
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class DenseSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_dense_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DenseSigmoid(
            event_shape=torch.Size(event_shape),
            n_dense_layers=n_dense_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_dense_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DenseSigmoid(
            event_shape=torch.Size(event_shape),
            n_dense_layers=n_dense_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DeepDenseSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 coupling_kwargs: dict = None,
                 **kwargs):
        if coupling_kwargs is None:
            coupling_kwargs = dict()
        coupling = make_coupling(event_shape, edge_list, **coupling_kwargs)
        transformer = DeepDenseSigmoid(
            event_shape=torch.Size((coupling.target_event_size,)),
            n_hidden_layers=n_hidden_layers
        )
        # Parameter order: [c1, c2, c3, c4, ..., ck] for all components
        # Each component has parameter order [a_unc, b, w_unc]
        conditioner_transform = FeedForward(
            input_event_shape=torch.Size((coupling.source_event_size,)),
            parameter_shape=torch.Size(transformer.parameter_shape),
            context_shape=context_shape,
            **kwargs
        )
        super().__init__(transformer, coupling, conditioner_transform)


class DeepDenseSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DeepDenseSigmoid(
            event_shape=torch.Size(event_shape),
            n_hidden_layers=n_hidden_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DeepDenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = 2,
                 **kwargs):
        transformer: ScalarTransformer = DeepDenseSigmoid(
            event_shape=torch.Size(event_shape),
            n_hidden_layers=n_hidden_layers
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


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


class AffineForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer: ScalarTransformer = Affine(event_shape=event_shape)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class RQSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class LRSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer: ScalarTransformer = LinearRational(event_shape=event_shape, n_bins=n_bins)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 **kwargs):
        transformer: ScalarTransformer = invert(Affine(event_shape=event_shape))
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class RQSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        assert n_bins >= 1
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, n_bins=n_bins)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class LRSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_bins: int = 8,
                 **kwargs):
        transformer: ScalarTransformer = LinearRational(event_shape=event_shape, n_bins=n_bins)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class UMNNMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: torch.Size,
                 context_shape: torch.Size = None,
                 n_hidden_layers: int = None,
                 hidden_dim: int = None,
                 **kwargs):
        transformer: ScalarTransformer = UnconstrainedMonotonicNeuralNetwork(
            event_shape=event_shape,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)
