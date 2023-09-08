from typing import Tuple, Union

import torch

from normalizing_flows.bijections.finite.base import Bijection


class Transformer(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, context=None)

    def inverse(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().inverse(x, context=None)

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
