import torch
import torch.nn as nn


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
