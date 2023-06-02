import torch

from src.bijections.finite.autoregressive.conditioners.base import Conditioner


class MaskedAutoregressive(Conditioner):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, transform, context: torch.Tensor = None) -> torch.Tensor:
        return transform(x, context=context)
