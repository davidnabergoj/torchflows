from src.bijections.finite.autoregressive.layers import (
    FeedForwardShiftCoupling,
    FeedForwardAffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive, FeedForwardRationalQuadraticSplineCoupling, SplineForwardMaskedAutoregressive,
    SplineInverseMaskedAutoregressive
)
from src.bijections.finite.base import BijectiveComposition
from src.bijections.finite.linear.permutation import Permutation


class NICE(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), FeedForwardShiftCoupling(n_dim, **kwargs)])
        super().__init__(bijections)


class RealNVP(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), FeedForwardAffineCoupling(n_dim, **kwargs)])
        super().__init__(bijections)


class MAF(BijectiveComposition):
    def __init__(self, n_dim, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), AffineForwardMaskedAutoregressive(n_dim, **kwargs)])
        super().__init__(bijections)


class IAF(BijectiveComposition):
    def __init__(self, n_dim, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), AffineInverseMaskedAutoregressive(n_dim, **kwargs)])
        super().__init__(bijections)


class CouplingRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), FeedForwardRationalQuadraticSplineCoupling(n_dim, **kwargs)])
        super().__init__(bijections)


class MaskedAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), SplineForwardMaskedAutoregressive(n_dim, **kwargs)])
        super().__init__(bijections)


class InverseAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim), SplineInverseMaskedAutoregressive(n_dim, **kwargs)])
        super().__init__(bijections)
