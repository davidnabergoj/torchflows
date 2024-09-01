from typing import Tuple, Union

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
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to Affine.
        """
        transformer = Affine(event_shape, **kwargs)
        super().__init__(transformer)


class ElementwiseInverseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseAffine.
        """
        transformer = InverseAffine(event_shape, **kwargs)
        super().__init__(transformer)


class ActNorm(ElementwiseInverseAffine):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ElementwiseInverseAffine.
        """
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
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to Scale.
        """
        transformer = Scale(event_shape, **kwargs)
        super().__init__(transformer)


class ElementwiseShift(ElementwiseBijection):
    def __init__(self, event_shape):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        """
        transformer = Shift(event_shape)
        super().__init__(transformer)


class ElementwiseRQSpline(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to RationalQuadratic.
        """
        transformer = RationalQuadratic(event_shape, **kwargs)
        super().__init__(transformer)


class AffineCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        if event_shape == (1,):
            raise ValueError
        super().__init__(event_shape, Affine, **kwargs)


class InverseAffineCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        if event_shape == (1,):
            raise ValueError
        super().__init__(event_shape, InverseAffine, **kwargs)


class ShiftCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        super().__init__(event_shape, Shift, **kwargs)


class LRSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        super().__init__(event_shape, LinearRational, **kwargs)


class RQSCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        super().__init__(event_shape, RationalQuadratic, **kwargs)


class DeepSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        super().__init__(event_shape, DeepSigmoid, **kwargs)


class DeepSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DeepSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DeepSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DeepSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class DenseSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DenseSigmoid, **kwargs)


class DenseSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param float percentage_global_parameters: percentage of transformer inputs to be learned globally instead of
         being predicted from the conditioner neural network.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DenseSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(
            event_shape,
            context_shape,
            transformer=transformer,
            **{
                **kwargs,
                **dict(percentage_global_parameters=percentage_global_parameters)
            }
        )


class DenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_transformer_layers: number of transformer layers.
        :param float percentage_global_parameters: percentage of transformer inputs to be learned globally instead of
         being predicted from the conditioner neural network.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DenseSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(
            event_shape,
            context_shape,
            transformer=transformer,
            **{
                **kwargs,
                **dict(percentage_global_parameters=percentage_global_parameters)
            }
        )


class DeepDenseSigmoidalCoupling(CouplingBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to CouplingBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DeepDenseSigmoid, **kwargs)


class DeepDenseSigmoidalInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_transformer_hidden_layers: number of transformer hidden layers.
        :param int n_transformer_dense_layers: number of transformer dense layers.
        :param int transformer_hidden_size: transformer hidden layer size.
        :param float percentage_global_parameters: percentage of transformer inputs to be learned globally instead of
         being predicted from the conditioner neural network.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DeepDenseSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(
            event_shape,
            context_shape,
            transformer=transformer,
            **{
                **kwargs,
                **dict(percentage_global_parameters=percentage_global_parameters)
            }
        )


class DeepDenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_transformer_hidden_layers: number of transformer hidden layers.
        :param int n_transformer_dense_layers: number of transformer dense layers.
        :param int transformer_hidden_size: transformer hidden layer size.
        :param float percentage_global_parameters: percentage of transformer inputs to be learned globally instead of
         being predicted from the conditioner neural network.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = DeepDenseSigmoid(
            event_shape=torch.Size(event_shape),
            **transformer_kwargs
        )
        super().__init__(
            event_shape,
            context_shape,
            transformer=transformer,
            **{
                **kwargs,
                **dict(percentage_global_parameters=percentage_global_parameters)
            }
        )


class LinearAffineCoupling(AffineCoupling):
    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to AffineCoupling.
        """
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearRQSCoupling(RQSCoupling):
    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to RQSCoupling.
        """
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearLRSCoupling(LRSCoupling):
    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to LRSCoupling.
        """
        super().__init__(event_shape, **kwargs, n_layers=1)


class LinearShiftCoupling(ShiftCoupling):
    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ShiftCoupling.
        """
        super().__init__(event_shape, **kwargs, n_layers=1)


class AffineForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = Affine(event_shape=event_shape, **transformer_kwargs)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class RQSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_bins: number of spline bins.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, **transformer_kwargs)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class LRSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_bins: number of spline bins.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = LinearRational(event_shape=event_shape, **transformer_kwargs)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = invert(Affine(event_shape=event_shape, **transformer_kwargs))
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class RQSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param int n_bins: number of spline bins.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = RationalQuadratic(event_shape=event_shape, **transformer_kwargs)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class LRSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param dict transformer_kwargs: keyword arguments to LinearRational.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = LinearRational(event_shape=event_shape, **transformer_kwargs)
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)


class UMNNMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 context_shape: Union[Tuple[int, ...], torch.Size] = None,
                 transformer_kwargs: dict = None,
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param Union[Tuple[int, ...], torch.Size] context_shape: shape of the context tensor.
        :param dict transformer_kwargs: keyword arguments to UnconstrainedMonotonicNeuralNetwork.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        transformer_kwargs = transformer_kwargs or {}
        transformer: ScalarTransformer = UnconstrainedMonotonicNeuralNetwork(
            event_shape=event_shape,
            **transformer_kwargs
        )
        super().__init__(event_shape, context_shape, transformer=transformer, **kwargs)
