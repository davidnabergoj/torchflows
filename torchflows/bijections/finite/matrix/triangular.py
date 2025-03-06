from typing import Union, Tuple

import torch
import math
import torch.nn as nn

from torchflows.bijections.finite.matrix.base import InvertibleMatrix


class NonTrainableLowerTriangularInvertibleMatrix(InvertibleMatrix):
    def __init__(self,
                 matrix: torch.Tensor,
                 **kwargs):
        # only works for tensors with one event dimension
        super().__init__(event_shape=(matrix.shape[0],), **kwargs)
        self.register_buffer('matrix', matrix)

    @property
    def unitriangular(self):
        return bool(torch.all(torch.diag(self.matrix) == 1))

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return torch.einsum('...ij,...j->...i', self.matrix, x_flat)

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b_flat_batch = b_flat.view(-1, b_flat.shape[-1])
        x_flat_batch = torch.linalg.solve_triangular(
            self.matrix,
            b_flat_batch.T.to(self.device_buffer.device),
            upper=False,
            unitriangular=self.unitriangular
        ).T
        return x_flat_batch.view_as(b_flat_batch)

    def log_det_project(self) -> torch.Tensor:
        return torch.diag(self.matrix).log().sum()


class LowerTriangularInvertibleMatrix(InvertibleMatrix):
    """
    Lower triangular matrix with strictly positive diagonal values.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 unitriangular: bool = False,
                 min_eigval: float = 1e-3,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.unitriangular = unitriangular
        self.min_eigval = min_eigval

        self.min_eigval = min_eigval
        self.log_min_eigval = math.log(min_eigval)

        self.off_diagonal_indices = torch.tril_indices(self.n_dim, self.n_dim, -1)
        self.register_parameter(
            'off_diagonal_elements',
            nn.Parameter(
                torch.randn((self.n_dim ** 2 - self.n_dim) // 2) / self.n_dim ** 2
            )
        )
        if not unitriangular:
            self.register_parameter('unc_diag_elements', nn.Parameter(torch.zeros(self.n_dim)))
            raise RuntimeError

    def constrain_diagonal_elements(self, u):
        return torch.exp(u) + self.min_eigval

    def _unconstrain_diagonal_elements(self, c):
        return torch.log(c - self.min_eigval)

    def compute_matrix(self):
        if self.unitriangular:
            mat = torch.eye(self.n_dim)
        else:
            mat = torch.diag(self.constrain_diagonal_elements(self.unc_diag_elements))
        mat[self.off_diagonal_indices[0], self.off_diagonal_indices[1]] = self.off_diagonal_elements
        return mat.to(self.device_buffer.device)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return torch.einsum('...ij,...j->...i', self.compute_matrix(), x_flat)

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b_flat_batch = b_flat.view(-1, b_flat.shape[-1])
        x_flat_batch = torch.linalg.solve_triangular(
            self.compute_matrix(),
            b_flat_batch.T.to(self.device_buffer.device),
            upper=False,
            unitriangular=self.unitriangular
        ).T
        return x_flat_batch.view_as(b_flat_batch)

    def log_det_project(self) -> torch.Tensor:
        return torch.logaddexp(
            self.unc_diag_elements,
            self.log_min_eigval * torch.ones_like(self.unc_diag_elements)
        ).sum()


class UpperTriangularInvertibleMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.lower = LowerTriangularInvertibleMatrix(event_shape=event_shape, **kwargs)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return torch.einsum('...ij,...j->...i', self.lower.compute_matrix().T, x_flat)

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b_flat_batch = b_flat.view(-1, b_flat.shape[-1])
        x_flat_batch = torch.linalg.solve_triangular(
            self.lower.compute_matrix().T,
            b_flat_batch.T.to(self.device_buffer.device),
            upper=True,
            unitriangular=self.lower.unitriangular
        ).T
        return x_flat_batch.view_as(b_flat_batch)

    def compute_matrix(self):
        return self.lower.compute_matrix().T

    def log_det_project(self) -> torch.Tensor:
        return self.lower.log_det_project()
