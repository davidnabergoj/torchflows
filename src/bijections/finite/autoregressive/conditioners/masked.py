import torch
import torch.nn as nn

from src.bijections.finite.autoregressive.conditioners.base import Conditioner


class MaskedAutoregressive(Conditioner):
    def __init__(self, transform: nn.Module, n_dim: int):
        super().__init__()
        self.transform = transform
        self.n_dim = n_dim

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.transform(x, context=context)
