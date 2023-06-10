import torch
import torch.nn as nn

from normalizing_flows.src.bijections import ConditionerTransform


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, transform: ConditionerTransform, context: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
