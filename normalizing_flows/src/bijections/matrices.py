import torch
import torch.nn as nn


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


class IdentityMatrix(Matrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim, **kwargs)

    def mat(self):
        return torch.eye(self.n_dim)

    def log_det(self):
        return torch.tensor(0.0)


class PositiveDiagonalMatrix(Matrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim, **kwargs)
        self.log_diag = nn.Parameter(torch.zeros(n_dim))

    def mat(self):
        return torch.diag(self.log_diag.exp())

    def log_det(self):
        return self.log_diag.sum()


class PermutationMatrix(Matrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim, **kwargs)
        self.perm = torch.randperm(self.n_dim)

    def mat(self):
        return torch.eye(self.n_dim)[self.perm]

    def log_det(self):
        return torch.tensor(0.0)
