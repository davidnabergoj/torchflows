from typing import Type, Union, Tuple, Optional

import torch

from torchflows.bijections.finite.autoregressive.layers import (
    ShiftCoupling,
    AffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    RQSCoupling,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive,
    InverseAffineCoupling,
    DeepSigmoidalCoupling,
    ElementwiseAffine,
    UMNNMaskedAutoregressive,
    LRSCoupling,
    LRSForwardMaskedAutoregressive,
    LRSInverseMaskedAutoregressive,
    DenseSigmoidalCoupling,
    DeepDenseSigmoidalCoupling, DeepSigmoidalInverseMaskedAutoregressive, DeepSigmoidalForwardMaskedAutoregressive,
    DenseSigmoidalInverseMaskedAutoregressive, DenseSigmoidalForwardMaskedAutoregressive,
    DeepDenseSigmoidalInverseMaskedAutoregressive, DeepDenseSigmoidalForwardMaskedAutoregressive, ActNorm
)
from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers_base import CouplingBijection, \
    MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection
from torchflows.bijections.finite.matrix.permutation import ReversePermutationMatrix


class AutoregressiveArchitecture(BijectiveComposition):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 base_bijection: Type[
                     Union[
                         CouplingBijection,
                         MaskedAutoregressiveBijection,
                         InverseMaskedAutoregressiveBijection
                     ]
                 ],
                 context_shape: Optional[Union[Tuple[int, ...], torch.Size, int]] = None,
                 n_layers: int = 2,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape, context_shape=context_shape)]
        for _ in range(n_layers):
            if 'edge_list' not in kwargs or kwargs['edge_list'] is None:
                bijections.append(ReversePermutationMatrix(event_shape=event_shape, context_shape=context_shape))
            bijections.append(base_bijection(event_shape=event_shape, context_shape=context_shape, **kwargs))
            bijections.append(ActNorm(event_shape=event_shape))
        bijections.append(ElementwiseAffine(event_shape=event_shape, context_shape=context_shape))
        bijections.append(ActNorm(event_shape=event_shape, context_shape=context_shape))
        super().__init__(bijections)


class NICE(AutoregressiveArchitecture):
    """Nonlinear independent components estimation (NICE) architecture.

    Reference: Dinh et al. "NICE: Non-linear Independent Components Estimation" (2015); https://arxiv.org/abs/1410.8516.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to ShiftCoupling.
        """
        super().__init__(event_shape, base_bijection=ShiftCoupling, **kwargs)


class RealNVP(AutoregressiveArchitecture):
    """Real non-volume-preserving (Real NVP) architecture.

    Reference: Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to :class:`~bijections.finite.autoregressive.layers.AffineCoupling`.
        """
        super().__init__(event_shape, base_bijection=AffineCoupling, **kwargs)


class InverseRealNVP(AutoregressiveArchitecture):
    """Inverse of the Real NVP architecture.

    Reference: Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to InverseAffineCoupling.
        """
        super().__init__(event_shape, base_bijection=InverseAffineCoupling, **kwargs)


class MAF(AutoregressiveArchitecture):
    """Masked autoregressive flow (MAF) architecture.

    Reference: Papamakarios et al. "Masked Autoregressive Flow for Density Estimation" (2018); https://arxiv.org/abs/1705.07057.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to AffineForwardMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=AffineForwardMaskedAutoregressive, **kwargs)


class IAF(AutoregressiveArchitecture):
    """Inverse autoregressive flow (IAF) architecture.

    Reference: Kingma et al. "Improving Variational Inference with Inverse Autoregressive Flow" (2017); https://arxiv.org/abs/1606.04934.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to AffineInverseMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=AffineInverseMaskedAutoregressive, **kwargs)


class CouplingRQNSF(AutoregressiveArchitecture):
    """Coupling rational quadratic neural spline flow (C-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to RQSCoupling.
        """
        super().__init__(event_shape, base_bijection=RQSCoupling, **kwargs)


class MaskedAutoregressiveRQNSF(AutoregressiveArchitecture):
    """Masked autoregressive rational quadratic neural spline flow (MA-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to RQSForwardMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=RQSForwardMaskedAutoregressive, **kwargs)


class CouplingLRS(AutoregressiveArchitecture):
    """Coupling linear rational spline (C-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to LRSCoupling.
        """
        super().__init__(event_shape, base_bijection=LRSCoupling, **kwargs)


class MaskedAutoregressiveLRS(AutoregressiveArchitecture):
    """Masked autoregressive linear rational spline (MA-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to LRSForwardMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=LRSForwardMaskedAutoregressive, **kwargs)


class InverseAutoregressiveRQNSF(AutoregressiveArchitecture):
    """Inverse autoregressive rational quadratic neural spline flow (IA-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to RQSInverseMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=RQSInverseMaskedAutoregressive, **kwargs)


class InverseAutoregressiveLRS(AutoregressiveArchitecture):
    """Inverse autoregressive linear rational spline (MA-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to LRSInverseMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=LRSInverseMaskedAutoregressive, **kwargs)


class CouplingDeepSF(AutoregressiveArchitecture):
    """Coupling deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to DeepSigmoidalCoupling.
        """
        super().__init__(event_shape, base_bijection=DeepSigmoidalCoupling, **kwargs)


class InverseAutoregressiveDeepSF(AutoregressiveArchitecture):
    """Inverse autoregressive deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to DeepSigmoidalInverseMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=DeepSigmoidalInverseMaskedAutoregressive, **kwargs)


class MaskedAutoregressiveDeepSF(AutoregressiveArchitecture):
    """Masked autoregressive deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to DeepSigmoidalForwardMaskedAutoregressive.
        """
        super().__init__(event_shape, base_bijection=DeepSigmoidalForwardMaskedAutoregressive, **kwargs)


class CouplingDenseSF(AutoregressiveArchitecture):
    """Coupling dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DenseSigmoidalCoupling.
        """
        super().__init__(
            event_shape,
            base_bijection=DenseSigmoidalCoupling,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class InverseAutoregressiveDenseSF(AutoregressiveArchitecture):
    """Inverse autoregressive dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DenseSigmoidalInverseMaskedAutoregressive.
        """
        super().__init__(
            event_shape,
            base_bijection=DenseSigmoidalInverseMaskedAutoregressive,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class MaskedAutoregressiveDenseSF(AutoregressiveArchitecture):
    """Masked autoregressive dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DenseSigmoidalForwardMaskedAutoregressive.
        """
        super().__init__(
            event_shape,
            base_bijection=DenseSigmoidalForwardMaskedAutoregressive,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class CouplingDeepDenseSF(AutoregressiveArchitecture):
    """Coupling deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DeepDenseSigmoidalCoupling.
        """
        super().__init__(
            event_shape,
            base_bijection=DeepDenseSigmoidalCoupling,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class InverseAutoregressiveDeepDenseSF(AutoregressiveArchitecture):
    """Inverse autoregressive deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DeepDenseSigmoidalInverseMaskedAutoregressive.
        """
        super().__init__(
            event_shape,
            base_bijection=DeepDenseSigmoidalInverseMaskedAutoregressive,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class MaskedAutoregressiveDeepDenseSF(AutoregressiveArchitecture):
    """Masked autoregressive deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size, int],
                 percentage_global_parameters: float = 0.8,
                 **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param percentage_global_parameters: percentage of transformer inputs to be learned globally instead of being
         predicted from the conditioner neural network.
        :param kwargs: keyword arguments to DeepDenseSigmoidalForwardMaskedAutoregressive.
        """
        super().__init__(
            event_shape,
            base_bijection=DeepDenseSigmoidalForwardMaskedAutoregressive,
            percentage_global_parameters=percentage_global_parameters,
            **kwargs
        )


class UMNNMAF(AutoregressiveArchitecture):
    """Unconstrained monotonic neural network masked autoregressive flow (UMNN-MAF) architecture.

    Reference: Wehenkel and Louppe "Unconstrained Monotonic Neural Networks" (2021); https://arxiv.org/abs/1908.05164.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size, int], **kwargs):
        """

        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments to UMNNMaskedAutoregressive.
        """
        super().__init__(
            event_shape,
            base_bijection=UMNNMaskedAutoregressive,
            **kwargs
        )
