import torch

from src.bijections.finite.autoregressive.layers import (
    ShiftCoupling,
    AffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    RQSCoupling,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive
)
from src.bijections.finite.base import BijectiveComposition
from src.bijections.finite.linear.permutation import Permutation


class NICE(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                ShiftCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class RealNVP(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class MAF(BijectiveComposition):
    def __init__(self, n_dim, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineForwardMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class IAF(BijectiveComposition):
    def __init__(self, n_dim, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                AffineInverseMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class CouplingRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSCoupling(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class MaskedAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSForwardMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)


class InverseAutoregressiveRQNSF(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10, **kwargs):
        event_shape = torch.Size((n_dim,))
        bijections = []
        for _ in range(n_layers):
            bijections.extend([
                Permutation(event_shape=event_shape),
                RQSInverseMaskedAutoregressive(event_shape=event_shape, **kwargs)
            ])
        super().__init__(bijections)
