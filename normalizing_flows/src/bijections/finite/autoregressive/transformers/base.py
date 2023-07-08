from typing import Tuple, Union

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__()
        self.event_shape = event_shape

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns transformed point and log Jacobian determinant of the transform
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns transformed point and log Jacobian determinant of the transform
        raise NotImplementedError