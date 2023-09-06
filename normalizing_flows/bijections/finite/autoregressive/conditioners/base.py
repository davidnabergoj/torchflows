import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, transform: ConditionerTransform, context: torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class NullConditioner(Conditioner):
    def __init__(self):
        # Each dimension affects only itself
        super().__init__()

    def forward(self, x: torch.Tensor, transform: ConditionerTransform, context: torch.Tensor = None) -> torch.Tensor:
        return transform(x, context=context).to(x)  # (*batch_shape, *event_shape, n_parameters)
