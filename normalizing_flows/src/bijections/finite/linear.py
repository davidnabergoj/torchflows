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


class Matrix(nn.Module):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        assert self.n_dim >= 2

    def mat(self):
        pass

    def log_det(self):
        pass

    def forward(self):
        return self.mat()


class LowerTriangularInvertibleMatrix(Matrix):
    def __init__(self, n_dim: int, unitriangular=False):
        super().__init__(n_dim)
        self.off_diagonal_elements = nn.Parameter(torch.randn((self.n_dim ** 2 - self.n_dim) // 2)) / self.n_dim ** 2
        if unitriangular:
            self.diagonal_elements = torch.ones(self.n_dim)
        else:
            self.diagonal_elements = nn.Parameter(torch.randn(self.n_dim))
        self.off_diagonal_indices = torch.tril_indices(self.n_dim, self.n_dim, -1)

    def mat(self):
        mat = torch.zeros(self.n_dim, self.n_dim)
        mat[range(self.n_dim), range(self.n_dim)] = self.diagonal_elements
        mat[self.off_diagonal_indices[0], self.off_diagonal_indices[1]] = self.off_diagonal_elements
        return mat

    def log_det(self):
        return torch.sum(torch.log(torch.abs(self.diagonal_elements)))


class UpperTriangularInvertibleMatrix(Matrix):
    def __init__(self, n_dim: int):
        super().__init__(n_dim)
        self.lower = LowerTriangularInvertibleMatrix(n_dim=n_dim)

    def mat(self):
        return self.lower().T

    def log_det(self):
        return -self.lower.log_det()


class HouseholderOrthogonalMatrix(Matrix):
    def __init__(self, n_dim: int, n_factors: int = None):
        super().__init__(n_dim=n_dim)
        if n_factors is None:
            n_factors = min(5, self.n_dim)
        assert 1 <= n_factors <= self.n_dim
        self.v = nn.Parameter(torch.randn(n_factors, self.n_dim) / self.n_dim ** 2 + torch.eye(n_factors, self.n_dim))
        self.tau = torch.full((n_factors,), fill_value=2.0)

    def mat(self):
        # TODO compute this more efficiently
        v_outer = torch.einsum('fi,fj->fij', self.v, self.v)
        v_norms_squared = torch.linalg.norm(self.v, dim=1).view(-1, 1, 1) ** 2
        h = (torch.eye(self.n_dim)[None] - 2 * (v_outer / v_norms_squared))
        return torch.linalg.multi_dot(list(h))

    def log_det(self):
        return torch.tensor(0.0)


class LowerTriangular(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        self.lower = LowerTriangularInvertibleMatrix(self.n_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        z = torch.einsum('ij,bj->bi', self.lower.mat(), x.view(-1, self.n_dim))
        log_det = torch.ones(size=batch_shape) * self.lower.log_det()
        z = torch.reshape(z, x.shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        x = torch.linalg.solve_triangular(
            self.lower.mat(),
            z.reshape(-1, self.n_dim).T,
            upper=False,
            unitriangular=False
        ).T
        log_det = -torch.ones(size=batch_shape) * self.lower.log_det()
        x = torch.reshape(x, z.shape)
        return x, log_det


class LU(Bijection):
    def __init__(self, event_shape: torch.Size):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        assert self.n_dim >= 2
        self.lower = LowerTriangularInvertibleMatrix(self.n_dim, unitriangular=True)
        self.upper = UpperTriangularInvertibleMatrix(self.n_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        xr = torch.reshape(x, (-1, self.n_dim))
        z = torch.einsum('ij,bj->bi', self.lower.mat() @ self.upper.mat(), xr)
        log_det = torch.ones(size=batch_shape) * self.upper.log_det()
        z = torch.reshape(z, x.shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        zr = torch.reshape(z, (-1, self.n_dim))
        x = torch.linalg.solve_triangular(
            self.upper.mat(),
            torch.linalg.solve_triangular(
                self.lower.mat(),
                zr.T,
                upper=False,
                unitriangular=True
            ),
            upper=True
        ).T
        log_det = -torch.ones(size=batch_shape) * self.upper.log_det()
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
        self.orthogonal = HouseholderOrthogonalMatrix(self.n_dim, n_factors)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        z = (self.orthogonal.mat() @ x.reshape(-1, self.n_dim).T).T.view_as(x)
        log_det = torch.zeros(size=batch_shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        x = (self.orthogonal.mat().T @ z.reshape(-1, self.n_dim).T).T.view_as(z)
        log_det = torch.zeros(size=batch_shape)
        return x, log_det


class QR(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.tensor(self.event_shape)))
        self.upper = UpperTriangularInvertibleMatrix(self.n_dim)
        self.orthogonal = HouseholderOrthogonalMatrix(self.n_dim, **kwargs)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        y = self.upper.mat() @ x.reshape(-1, self.n_dim).T
        z = self.orthogonal.mat() @ y
        z = z.T.reshape(x.shape)
        log_det = torch.ones(size=batch_shape) * self.upper.log_det()
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        y = self.orthogonal.mat().T @ z.reshape(-1, self.n_dim).T
        x = torch.linalg.solve_triangular(self.upper.mat(), y, upper=True, unitriangular=False)
        x = x.T.reshape(z.shape)
        log_det = -torch.ones(size=batch_shape) * self.upper.log_det()
        return x, log_det
