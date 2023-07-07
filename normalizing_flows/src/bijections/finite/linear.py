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


class LowerTriangular(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        assert self.n_dim >= 2
        self.elements = nn.Parameter(
            torch.randn((self.n_dim ** 2 + self.n_dim) // 2)
        )
        self.indices = torch.tril_indices(self.n_dim, self.n_dim)

    def construct_mat(self):
        mat = torch.zeros(self.n_dim, self.n_dim)
        mat[self.indices[0], self.indices[1]] = self.elements
        return mat

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = self.construct_mat()
        batch_shape = get_batch_shape(x, self.event_shape)
        z = torch.einsum('ij,bj->bi', lower, x.view(-1, self.n_dim))
        fill_value = torch.sum(torch.log(torch.abs(lower[range(self.n_dim), range(self.n_dim)])))
        log_det = torch.ones(size=batch_shape) * fill_value
        z = torch.reshape(z, x.shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = self.construct_mat()
        batch_shape = get_batch_shape(z, self.event_shape)
        x = torch.linalg.solve_triangular(
            lower,
            z.reshape(-1, self.n_dim).T,
            upper=False,
            unitriangular=False
        ).T
        fill_value = -torch.sum(torch.log(torch.abs(lower[range(self.n_dim), range(self.n_dim)])))
        log_det = torch.ones(size=batch_shape) * fill_value
        x = torch.reshape(x, z.shape)
        return x, log_det


class LU(Bijection):
    def __init__(self, event_shape: torch.Size):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        assert self.n_dim >= 2
        self.lower_elements = nn.Parameter(torch.randn((self.n_dim ** 2 - self.n_dim) // 2) / self.n_dim ** 2)
        self.upper_elements = nn.Parameter(torch.randn((self.n_dim ** 2 + self.n_dim) // 2) / self.n_dim ** 2)

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
        x = torch.linalg.solve_triangular(
            upper,
            torch.linalg.solve_triangular(
                lower,
                zr.T,
                upper=False,
                unitriangular=True
            ),
            upper=True
        ).T
        # mat = lower @ upper
        # mat_inv = torch.linalg.inv(mat)
        # x = torch.einsum('ij,bj->bi', mat_inv, zr)
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


class HouseholderOrthogonal(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_factors: int = None):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        assert self.n_dim >= 2
        if n_factors is None:
            n_factors = min(5, self.n_dim)
        assert 1 <= n_factors <= self.n_dim
        self.v = nn.Parameter(torch.randn(n_factors, self.n_dim))
        self.tau = torch.full((n_factors,), fill_value=2.0)

    def construct_mat(self):
        # TODO compute this more efficiently
        v_outer = torch.einsum('fi,fj->fij', self.v, self.v)
        v_norms_squared = torch.linalg.norm(self.v, dim=1).view(-1, 1, 1) ** 2
        h = (torch.eye(self.n_dim)[None] - 2 * (v_outer / v_norms_squared))
        return torch.linalg.multi_dot(list(h))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        orthogonal = self.construct_mat()
        batch_shape = get_batch_shape(x, self.event_shape)
        z = (orthogonal @ x.reshape(-1, self.n_dim).T).T.view_as(x)
        log_det = torch.zeros(size=batch_shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        orthogonal = self.construct_mat().T
        batch_shape = get_batch_shape(z, self.event_shape)
        x = (orthogonal @ z.reshape(-1, self.n_dim).T).T.view_as(z)
        log_det = torch.zeros(size=batch_shape)
        return x, log_det


class QR(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def construct_orthogonal(self):
        lower = torch.eye(self.n_dim)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_elements
        return lower

    def construct_upper(self):
        upper = torch.zeros(self.n_dim, self.n_dim)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_elements
        return upper
