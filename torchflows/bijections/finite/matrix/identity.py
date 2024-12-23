from typing import Union, Tuple

import torch
from torchflows.bijections.finite.matrix.base import InvertibleMatrix


class IdentityMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return x_flat

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return b_flat

    def log_det_project(self):
        return torch.zeros(1).to(self.device_buffer.device)