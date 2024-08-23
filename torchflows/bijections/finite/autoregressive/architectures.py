from typing import Tuple, List, Type, Union

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
    DeepDenseSigmoidalInverseMaskedAutoregressive, DeepDenseSigmoidalForwardMaskedAutoregressive
)
from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.layers_base import CouplingBijection, \
    MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection
from torchflows.bijections.finite.linear import ReversePermutation


def make_basic_layers(base_bijection: Type[
    Union[CouplingBijection, MaskedAutoregressiveBijection, InverseMaskedAutoregressiveBijection]],
                      event_shape,
                      n_layers: int = 2,
                      edge_list: List[Tuple[int, int]] = None):
    """
    Returns a list of bijections for transformations of vectors.
    """
    bijections = [ElementwiseAffine(event_shape=event_shape)]
    for _ in range(n_layers):
        if edge_list is None:
            bijections.append(ReversePermutation(event_shape=event_shape))
        bijections.append(base_bijection(event_shape=event_shape, edge_list=edge_list))
    bijections.append(ElementwiseAffine(event_shape=event_shape))
    return bijections


class NICE(BijectiveComposition):
    """Nonlinear independent components estimation (NICE) architecture.

    Reference: Dinh et al. "NICE: Non-linear Independent Components Estimation" (2015); https://arxiv.org/abs/1410.8516.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(ShiftCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class RealNVP(BijectiveComposition):
    """Real non-volume-preserving (Real NVP) architecture.

    Reference: Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(AffineCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class InverseRealNVP(BijectiveComposition):
    """Inverse of the Real NVP architecture.

    Reference: Dinh et al. "Density estimation using Real NVP" (2017); https://arxiv.org/abs/1605.08803.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(InverseAffineCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MAF(BijectiveComposition):
    """Masked autoregressive flow (MAF) architecture.

    Reference: Papamakarios et al. "Masked Autoregressive Flow for Density Estimation" (2018); https://arxiv.org/abs/1705.07057.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(AffineForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class IAF(BijectiveComposition):
    """Inverse autoregressive flow (IAF) architecture.

    Reference: Kingma et al. "Improving Variational Inference with Inverse Autoregressive Flow" (2017); https://arxiv.org/abs/1606.04934.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(AffineInverseMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingRQNSF(BijectiveComposition):
    """Coupling rational quadratic neural spline flow (C-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(RQSCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveRQNSF(BijectiveComposition):
    """Masked autoregressive rational quadratic neural spline flow (MA-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(RQSForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingLRS(BijectiveComposition):
    """Coupling linear rational spline (C-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(LRSCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveLRS(BijectiveComposition):
    """Masked autoregressive linear rational spline (MA-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(LRSForwardMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveRQNSF(BijectiveComposition):
    """Inverse autoregressive rational quadratic neural spline flow (IA-RQNSF) architecture.

    Reference: Durkan et al. "Neural Spline Flows" (2019); https://arxiv.org/abs/1906.04032.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(RQSInverseMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveLRS(BijectiveComposition):
    """Inverse autoregressive linear rational spline (MA-LRS) architecture.

    Reference: Dolatabadi et al. "Invertible Generative Modeling using Linear Rational Splines" (2020); https://arxiv.org/abs/2001.05168.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(LRSInverseMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingDeepSF(BijectiveComposition):
    """Coupling deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepSigmoidalCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveDeepSF(BijectiveComposition):
    """Inverse autoregressive deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepSigmoidalInverseMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveDeepSF(BijectiveComposition):
    """Masked autoregressive deep sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepSigmoidalForwardMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingDenseSF(BijectiveComposition):
    """Coupling dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DenseSigmoidalCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveDenseSF(BijectiveComposition):
    """Inverse autoregressive dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DenseSigmoidalInverseMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveDenseSF(BijectiveComposition):
    """Masked autoregressive dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DenseSigmoidalForwardMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class CouplingDeepDenseSF(BijectiveComposition):
    """Coupling deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepDenseSigmoidalCoupling, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveDeepDenseSF(BijectiveComposition):
    """Inverse autoregressive deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepDenseSigmoidalInverseMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveDeepDenseSF(BijectiveComposition):
    """Masked autoregressive deep-dense sigmoidal flow architecture.

    Reference: Huang et al. "Neural Autoregressive Flows" (2018); https://arxiv.org/abs/1804.00779.
    """

    def __init__(self,
                 event_shape,
                 n_layers: int = 2,
                 edge_list: List[Tuple[int, int]] = None,
                 **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(DeepDenseSigmoidalForwardMaskedAutoregressive, event_shape, n_layers, edge_list)
        super().__init__(event_shape, bijections, **kwargs)


class UMNNMAF(BijectiveComposition):
    """Unconstrained monotonic neural network masked autoregressive flow (UMNN-MAF) architecture.

    Reference: Wehenkel and Louppe "Unconstrained Monotonic Neural Networks" (2021); https://arxiv.org/abs/1908.05164.
    """

    def __init__(self, event_shape, n_layers: int = 1, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = make_basic_layers(UMNNMaskedAutoregressive, event_shape, n_layers)
        super().__init__(event_shape, bijections, **kwargs)
