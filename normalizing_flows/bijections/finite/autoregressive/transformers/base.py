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

    @property
    def n_parameters(self) -> int:
        """
        Number of parameters that parametrize this transformer. Example: rational quadratic splines require 3*b-1 where
        b is the number of bins. An affine transformation requires 2 (typically corresponding to the unconstrained scale
        and shift).
        """
        raise NotImplementedError

    @property
    def default_parameters(self) -> torch.Tensor:
        """
        Set of parameters which ensures an identity transformation.
        """
        raise NotImplementedError


class Inverse(Transformer):
    def __init__(self, transformer: Transformer):
        super().__init__(transformer.event_shape)
        self.base_transformer = transformer

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_transformer.inverse(x, h)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_transformer.forward(z, h)
