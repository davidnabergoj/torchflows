import torch
import torch.nn as nn

from typing import Tuple, Union

from normalizing_flows.src.bijections.finite.base import Bijection
from normalizing_flows.src.utils import get_batch_shape


class Permutation(Bijection):
    def __init__(self, event_shape):
        super().__init__(event_shape=event_shape)
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


class LU(Bijection):
    def __init__(self, event_shape: torch.Size):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        assert self.n_dim >= 2
        self.lower_elements = nn.Parameter(torch.zeros((self.n_dim ** 2 - self.n_dim) // 2))
        self.upper_elements = nn.Parameter(torch.randn((self.n_dim ** 2 + self.n_dim) // 2))

        self.lower_indices = torch.tril_indices(self.n_dim, self.n_dim, -1)
        self.upper_indices = torch.triu_indices(self.n_dim, self.n_dim)

    def construct_lower(self):
        lower = torch.eye(self.n_dim)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_elements
        return lower

    def construct_upper(self):
        upper = torch.zeros(self.n_dim, self.n_dim)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_elements
        return upper

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = self.construct_lower()
        upper = self.construct_upper()
        mat = lower @ upper
        batch_shape = get_batch_shape(x, self.event_shape)
        xr = torch.reshape(x, (-1, self.n_dim))
        z = torch.einsum('ij,bj->bi', mat, xr)
        fill_value = torch.sum(torch.log(torch.abs(upper[range(self.n_dim), range(self.n_dim)])))
        log_det = torch.ones(size=batch_shape) * fill_value
        z = torch.reshape(z, x.shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = self.construct_lower()
        upper = self.construct_upper()
        batch_shape = get_batch_shape(z, self.event_shape)
        zr = torch.reshape(z, (-1, self.n_dim))
        mat = lower @ upper
        mat_inv = torch.linalg.inv(mat)
        x = torch.einsum('ij,bj->bi', mat_inv, zr)
        fill_value = -torch.sum(torch.log(torch.abs(upper[range(self.n_dim), range(self.n_dim)])))
        log_det = torch.ones(size=batch_shape) * fill_value
        x = torch.reshape(x, z.shape)
        return x, log_det


class InverseLU(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.lu = LU(event_shape=event_shape)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lu.inverse(x, context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lu.forward(z, context)
