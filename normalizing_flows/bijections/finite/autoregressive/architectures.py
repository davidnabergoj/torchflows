import torch

from normalizing_flows.bijections.finite.autoregressive.layers import (
    ShiftCoupling,
    AffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    RQSCoupling,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive,
    InverseAffineCoupling, DSCoupling, InverseDSCoupling,
    UMNNForwardMaskedAutoregressive, ElementwiseAffine
)
from normalizing_flows.bijections.finite.base import BijectiveComposition
from normalizing_flows.bijections.finite.linear import Permutation


class NICE(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                ShiftCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class RealNVP(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class InverseRealNVP(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                InverseAffineCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class MAF(BijectiveComposition):
    """
    Expressive bijection with slightly unstable inverse due to autoregressive formulation.
    """
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineForwardMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class IAF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineInverseMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class CouplingRQNSF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class MaskedAutoregressiveRQNSF(BijectiveComposition):
    """
    Expressive bijection with unstable inverse due to autoregressive formulation.
    """
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSForwardMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class InverseAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSInverseMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class CouplingDSF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                DSCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class InverseCouplingDSF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                InverseDSCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)


class UMNNMAF(BijectiveComposition):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                UMNNMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(event_shape, bijections)
