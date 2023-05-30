import torch
from typing import Tuple

from src.bijections.finite.base import Bijection


class Permutation(Bijection):
    def __init__(self, n_dim: int):
        super().__init__()
        self.forward_permutation = torch.randperm(n_dim)
        self.inverse_permutation = torch.empty_like(self.forward_permutation)
        self.inverse_permutation[self.forward_permutation] = torch.arange(n_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x[:, self.forward_permutation]
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z[:, self.inverse_permutation]
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x, log_det
