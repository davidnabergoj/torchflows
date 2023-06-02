import torch
from typing import Tuple

from src.bijections.finite.base import Bijection
from src.utils import get_batch_shape


class Permutation(Bijection):
    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape
        n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        self.forward_permutation = torch.randperm(n_dim)
        self.inverse_permutation = torch.empty_like(self.forward_permutation)
        self.inverse_permutation[self.forward_permutation] = torch.arange(n_dim)

    def forward(self, x, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        z = x.view(*batch_shape, -1)[..., self.forward_permutation].view_as(x)
        log_det = torch.zeros(*batch_shape, device=x.device)
        return z, log_det

    def inverse(self, z, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        x = z.view(*batch_shape, -1)[..., self.inverse_permutation].view_as(z)
        log_det = torch.zeros(*batch_shape, device=z.device)
        return x, log_det
