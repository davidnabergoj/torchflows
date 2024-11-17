import torch
import torch.nn as nn


class InvertibleMatrix(nn.Module):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__()
        self.n_dim = n_dim

    def mat(self):
        raise NotImplementedError

    def log_det(self):
        raise NotImplementedError

    def project(self, x):
        # Compute z = mat @ x
        return torch.einsum('ij,...j->...i', self.mat(), x)

    def solve(self, z):
        # Compute x = mat^-1 z
        raise NotImplementedError


class LowerTriangularInvertibleMatrix(InvertibleMatrix):
    """
    Lower triangular matrix with strictly positive diagonal values.
    """

    def __init__(self, n_dim: int, unitriangular: bool = False, min_eigval: float = 1e-3):
        """

        :param n_dim:
        :param unitriangular:
        :param min_eigval: minimum eigenvalue. This is added to
        """
        super().__init__(n_dim)
        self.unitriangular = unitriangular

        n_off_diagonal_elements = (self.n_dim ** 2 - self.n_dim) // 2
        initial_off_diagonal_elements = torch.randn(n_off_diagonal_elements) / self.n_dim ** 2
        self.off_diagonal_elements = nn.Parameter(initial_off_diagonal_elements)
        if unitriangular:
            self.unc_diagonal_elements = None
        else:
            self.unc_diagonal_elements = nn.Parameter(torch.zeros(self.n_dim))
        self.off_diagonal_indices = torch.tril_indices(self.n_dim, self.n_dim, -1)
        self.min_eigval = min_eigval

        self.register_buffer('mat_zeros', torch.zeros(size=(self.n_dim, self.n_dim)))

    def mat(self):
        mat = self.mat_zeros
        mat[range(self.n_dim), range(self.n_dim)] = self.compute_diagonal_elements()
        mat[self.off_diagonal_indices[0], self.off_diagonal_indices[1]] = self.off_diagonal_elements
        return mat

    def compute_diagonal_elements(self):
        if self.unitriangular:
            return torch.ones(self.n_dim)
        else:
            return torch.exp(self.unc_diagonal_elements) + self.min_eigval

    def log_det(self):
        return torch.sum(torch.log(self.compute_diagonal_elements()))

    def solve(self, z):
        return torch.linalg.solve_triangular(self.mat(), z.T, upper=False, unitriangular=self.unitriangular).T


class UpperTriangularInvertibleMatrix(InvertibleMatrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim)
        self.lower = LowerTriangularInvertibleMatrix(n_dim=n_dim, **kwargs)

    def mat(self):
        return self.lower.mat().T

    def log_det(self):
        return self.lower.log_det()

    def solve(self, z):
        return torch.linalg.solve_triangular(self.mat(), z.T, upper=True, unitriangular=self.lower.unitriangular).T


class HouseholderOrthogonalMatrix(InvertibleMatrix):
    # TODO more efficient project and solve?
    def __init__(self, n_dim: int, n_factors: int = None):
        super().__init__(n_dim=n_dim)
        if n_factors is None:
            n_factors = min(5, self.n_dim)
        assert 1 <= n_factors <= self.n_dim
        self.v = nn.Parameter(torch.randn(n_factors, self.n_dim) / self.n_dim ** 2 + torch.eye(n_factors, self.n_dim))
        self.tau = torch.full((n_factors,), fill_value=2.0)

    def mat(self):
        v_outer = torch.einsum('fi,fj->fij', self.v, self.v)
        v_norms_squared = torch.linalg.norm(self.v, dim=1).view(-1, 1, 1) ** 2
        h = (torch.eye(self.n_dim)[None].to(v_outer) - 2 * (v_outer / v_norms_squared))
        return torch.linalg.multi_dot(list(h))

    def log_det(self):
        return 0.0

    def solve(self, z):
        return (self.mat().T @ z.T).T


class IdentityMatrix(InvertibleMatrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim, **kwargs)
        self.register_buffer('_mat', torch.eye(self.n_dim))

    def mat(self):
        return self._mat

    def log_det(self):
        return 0.0

    def project(self, x):
        return x

    def solve(self, z):
        return z


class PermutationMatrix(InvertibleMatrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim, **kwargs)
        self.forward_permutation = torch.randperm(n_dim)
        self.inverse_permutation = torch.empty_like(self.forward_permutation)
        self.inverse_permutation[self.forward_permutation] = torch.arange(n_dim)

    def mat(self):
        return torch.eye(self.n_dim)[self.forward_permutation]

    def log_det(self):
        return 0.0

    def project(self, x):
        return x[..., self.forward_permutation]

    def solve(self, z):
        return z[..., self.inverse_permutation]


class LUMatrix(InvertibleMatrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim)
        self.lower = LowerTriangularInvertibleMatrix(n_dim, unitriangular=True, **kwargs)
        self.upper = UpperTriangularInvertibleMatrix(n_dim, **kwargs)

    def mat(self):
        return self.lower.mat() @ self.upper.mat()

    def log_det(self):
        return self.upper.log_det()

    def solve(self, z):
        return self.upper.solve(self.lower.solve(z))


class QRMatrix(InvertibleMatrix):
    def __init__(self, n_dim: int, **kwargs):
        super().__init__(n_dim)
        self.orthogonal = HouseholderOrthogonalMatrix(self.n_dim, **kwargs)
        self.upper = UpperTriangularInvertibleMatrix(n_dim, **kwargs)

    def mat(self):
        return self.orthogonal.mat() @ self.upper.mat()

    def solve(self, z):
        return self.upper.solve(self.orthogonal.solve(z))

    def log_det(self):
        return self.upper.log_det()
