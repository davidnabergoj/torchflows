import torch
from src.bijections.finite.autoregressive.layers import FeedForwardAffineCoupling
from src.bijections.finite.base import BijectiveComposition
from src.bijections.finite.linear.permutation import Permutation


class RealNVP(BijectiveComposition):
    def __init__(self, n_dim: int, n_layers: int = 10):
        bijections = []
        for _ in range(n_layers):
            bijections.extend([Permutation(n_dim=n_dim), FeedForwardAffineCoupling(n_dim)])
        super().__init__(bijections)
