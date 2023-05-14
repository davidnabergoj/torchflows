import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer


class Affine(Transformer):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(h[..., 0])
        beta = h[..., 1]
        return alpha * x + beta

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(h[..., 0])
        beta = h[..., 1]
        return (z - beta) / alpha
