from typing import Tuple, Union

import torch

from torchflows.bijections.finite.autoregressive.layers_base import MaskedAutoregressiveBijection, \
    InverseMaskedAutoregressiveBijection, ElementwiseBijection, CouplingBijection
from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Scale, Affine, Shift, InverseAffine
from torchflows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network import \
    UnconstrainedMonotonicNeuralNetwork
from torchflows.bijections.finite.autoregressive.transformers.spline.linear_rational import LinearRational
from torchflows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid import (
    DeepSigmoid,
    DenseSigmoid,
    DeepDenseSigmoid
)


class ElementwiseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ElementwiseBijection.
        """
        super().__init__(event_shape, Affine, **kwargs)


class ElementwiseInverseAffine(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ElementwiseBijection.
        """
        super().__init__(event_shape, InverseAffine, **kwargs)


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
        :param kwargs: keyword arguments to ElementwiseBijection.
        """
        super().__init__(event_shape, Scale, **kwargs)


class ElementwiseShift(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ElementwiseBijection.
        """
        super().__init__(event_shape, Shift, **kwargs)


class ElementwiseRQSpline(ElementwiseBijection):
    def __init__(self, event_shape, **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ElementwiseBijection.
        """
        super().__init__(event_shape, RationalQuadratic, **kwargs)


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
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, DeepSigmoid, **kwargs)


class DeepSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, DeepSigmoid, **kwargs)


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
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DenseSigmoid, **kwargs)


class DenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DenseSigmoid, **kwargs)


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
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DeepDenseSigmoid, **kwargs)


class DeepDenseSigmoidalForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        if 'conditioner_kwargs' not in kwargs:
            kwargs['conditioner_kwargs'] = {}
        if 'percentage_global_parameters' not in kwargs['conditioner_kwargs']:
            kwargs['conditioner_kwargs']['percentage_global_parameters'] = 0.8
        super().__init__(event_shape, DeepDenseSigmoid, **kwargs)


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
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, Affine, **kwargs)


class RQSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, RationalQuadratic, **kwargs)


class LRSForwardMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, LinearRational, **kwargs)


class AffineInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, InverseAffine, **kwargs)


class RQSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, RationalQuadratic, **kwargs)


class LRSInverseMaskedAutoregressive(InverseMaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseMaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, LinearRational, **kwargs)


class UMNNMaskedAutoregressive(MaskedAutoregressiveBijection):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 **kwargs):
        """

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to MaskedAutoregressiveBijection.
        """
        super().__init__(event_shape, UnconstrainedMonotonicNeuralNetwork, **kwargs)
