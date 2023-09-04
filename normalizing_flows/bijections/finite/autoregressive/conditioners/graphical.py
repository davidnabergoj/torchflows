import torch

from normalizing_flows.bijections.finite.autoregressive.conditioners.base import Conditioner
from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform


class GraphicalConditioner(Conditioner):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, transform: ConditionerTransform, context: torch.Tensor = None) -> torch.Tensor:
        pass
