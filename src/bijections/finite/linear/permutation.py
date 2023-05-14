import torch
from typing import Tuple

from src.bijections.finite.base import Bijection


class Permutation(Bijection):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
