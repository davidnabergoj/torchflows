import torch

from normalizing_flows.src.bijections import Conditioner


class MaskedAutoregressive(Conditioner):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, transform, context: torch.Tensor = None) -> torch.Tensor:
        return transform(x, context=context)
