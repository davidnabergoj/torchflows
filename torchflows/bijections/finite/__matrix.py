# import math
# from typing import Union, Tuple
#
# import torch
# import torch.nn as nn
#
# from torchflows.bijections.base import Bijection
# from torchflows.utils import get_batch_shape
#
#
# class InvertibleMatrix(Bijection):
#     """
#     Invertible matrix bijection (currently ignores context).
#     """
#
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.register_buffer('device_buffer', torch.zeros(1))
#
#     def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_shape = get_batch_shape(x, self.event_shape)
#         x_flat = x.view(*batch_shape, -1)
#         context_flat = context.view(*batch_shape, -1) if context is not None else None
#         z_flat = self.project_flat(x_flat, context_flat)
#         z = z_flat.view_as(x)
#         log_det = self.log_det_project()[[None] * len(batch_shape)].repeat(*batch_shape)
#         return z, log_det
#
#     def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_shape = get_batch_shape(z, self.event_shape)
#         z_flat = z.view(*batch_shape, -1)
#         context_flat = context.view(*batch_shape, -1) if context is not None else None
#         x_flat = self.solve_flat(z_flat, context_flat)
#         x = x_flat.view_as(z)
#         log_det = -self.log_det_project()[[None] * len(batch_shape)].repeat(*batch_shape)
#         return x, log_det
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         raise NotImplementedError
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         """
#         Find x in Ax = b where b is given and A is this matrix.
#
#         :param b_flat: shift tensor with shape (self.n_dim,)
#         :param context:
#         :return:
#         """
#         raise NotImplementedError
#
#     def log_det_project(self) -> torch.Tensor:
#         """
#
#         :return: log abs det jac of f where f(x) = Ax and A is this matrix.
#         """
#         raise NotImplementedError
#
#
# class LowerTriangularInvertibleMatrix(InvertibleMatrix):
#     """
#     Lower triangular matrix with strictly positive diagonal values.
#     """
#
#     def __init__(self,
#                  event_shape: Union[torch.Size, Tuple[int, ...]],
#                  unitriangular: bool = False,
#                  min_eigval: float = 1e-3,
#                  **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.unitriangular = unitriangular
#         self.min_eigval = min_eigval
#
#         self.min_eigval = min_eigval
#         self.log_min_eigval = math.log(min_eigval)
#
#         self.off_diagonal_indices = torch.tril_indices(self.n_dim, self.n_dim, -1)
#         self.register_parameter(
#             'off_diagonal_elements',
#             nn.Parameter(
#                 torch.randn((self.n_dim ** 2 - self.n_dim) // 2) / self.n_dim ** 2
#             )
#         )
#         if not unitriangular:
#             self.register_parameter('unc_diagonal_elements', nn.Parameter(torch.zeros(self.n_dim)))
#
#     def compute_tril_matrix(self):
#         if self.unitriangular:
#             mat = torch.eye(self.n_dim)
#         else:
#             mat = torch.diag(torch.exp(self.unc_diagonal_elements) + self.min_eigval)
#         mat[self.off_diagonal_indices[0], self.off_diagonal_indices[1]] = self.off_diagonal_elements
#         return mat.to(self.device_buffer.device)
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         return torch.einsum('...ij,...j->...i', self.compute_tril_matrix(), x_flat)
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         return torch.linalg.solve_triangular(
#             self.compute_tril_matrix(),
#             b_flat[None].T.to(self.device_buffer.device),
#             upper=False,
#             unitriangular=self.unitriangular
#         ).T
#
#     def log_det_project(self) -> torch.Tensor:
#         return torch.logaddexp(
#             self.unc_diagonal_elements,
#             self.logmin_eigval * torch.ones_like(self.unc_diag_elements)
#         ).sum()
#
#
# class UpperTriangularInvertibleMatrix(InvertibleMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.lower = LowerTriangularInvertibleMatrix(event_shape=event_shape, **kwargs)
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         return torch.einsum('...ij,...j->...i', self.lower.compute_tril_matrix().T, x_flat)
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         return torch.linalg.solve_triangular(
#             self.lower.compute_tril_matrix().T,
#             b_flat[None].T.to(self.device_buffer.device),
#             upper=True,
#             unitriangular=self.unitriangular
#         ).T
#
#     def log_det_project(self) -> torch.Tensor:
#         return self.lower.log_det_project()
#
#
# class HouseholderOrthogonalMatrix(InvertibleMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_factors: int = None, **kwargs):
#         super().__init__(event_shape, **kwargs)
#         if n_factors is None:
#             n_factors = min(5, self.n_dim)
#         assert 1 <= n_factors <= self.n_dim
#
#         self.v = nn.Parameter(torch.randn(n_factors, self.n_dim) / self.n_dim ** 2 + torch.eye(n_factors, self.n_dim))
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         batch_shape = x_flat.shape[:-1]
#         z_flat = x_flat.clone()  # (*batch_shape, self.n_dim)
#         for i in range(self.v.shape[0]):  # Apply each Householder transformation in reverse order
#             v = self.v[i]  # (self.n_dim,)
#             alpha = 2 * torch.einsum('i,...i->...', v, z_flat)[..., None]  # (*batch_shape, 1)
#             v = v[[None] * len(batch_shape)]  # (1, ..., 1, self.n_dim) with len(v.shape) == len(batch_shape) + 1
#             z_flat = z_flat - alpha * v
#         return z_flat
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         # Same code as project, just the reverse matrix order
#         batch_shape = b_flat.shape[:-1]
#         x_flat = b_flat.clone()  # (*batch_shape, self.n_dim)
#         for i in range(self.v.shape[0] - 1, -1, -1):  # Apply each Householder transformation in reverse order
#             v = self.v[i]  # (self.n_dim,)
#             alpha = 2 * torch.einsum('i,...i->...', v, x_flat)[..., None]  # (*batch_shape, 1)
#             v = v[[None] * len(batch_shape)]  # (1, ..., 1, self.n_dim) with len(v.shape) == len(batch_shape) + 1
#             x_flat = x_flat - alpha * v
#         return x_flat
#
#     def log_det_project(self) -> torch.Tensor:
#         return torch.tensor(0.0).to(self.device_buffer.device)
#
#
# class IdentityMatrix(InvertibleMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         super().__init__(event_shape, **kwargs)
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         return x_flat
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         return b_flat
#
#     def log_det_project(self):
#         return torch.tensor(0.0).to(self.device_buffer.device)
#
#
# class PermutationMatrix(InvertibleMatrix):
#     def __init__(self,
#                  event_shape: Union[torch.Size, Tuple[int, ...]],
#                  forward_permutation: torch.Tensor,
#                  **kwargs):
#         super().__init__(event_shape, **kwargs)
#         assert forward_permutation.shape == event_shape
#         self.forward_permutation = forward_permutation.view(-1)
#         self.inverse_permutation = torch.empty_like(self.forward_permutation)
#         self.inverse_permutation[self.forward_permutation] = torch.arange(self.n_dim)
#
#
# class RandomPermutationMatrix(PermutationMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         n_dim = int(torch.prod(torch.as_tensor(event_shape)))
#         super().__init__(event_shape, forward_permutation=torch.randperm(n_dim).view(*event_shape), **kwargs)
#
#
# class ReversePermutationMatrix(PermutationMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         n_dim = int(torch.prod(torch.as_tensor(event_shape)))
#         super().__init__(event_shape, forward_permutation=torch.arange(n_dim)[::-1].view(*event_shape), **kwargs)
#
#
# class LUMatrix(InvertibleMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.lower = LowerTriangularInvertibleMatrix(self.n_dim, unitriangular=True, **kwargs)
#         self.upper = UpperTriangularInvertibleMatrix(self.n_dim, **kwargs)
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         return self.lower.project_flat(self.upper.project_flat(x_flat))
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         return self.upper.solve_flat(self.lower.solve_flat(b_flat))
#
#     def log_det_project(self):
#         return self.upper.logdet_project()
#
#
# class QRMatrix(InvertibleMatrix):
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.orthogonal = HouseholderOrthogonalMatrix(self.n_dim, **kwargs)
#         self.upper = UpperTriangularInvertibleMatrix(self.n_dim, **kwargs)
#
#     def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
#         return self.orthogonal.project_flat(self.upper.project_flat(x_flat))
#
#     def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
#         w_flat = self.orthogonal.project_inverse_flat(b_flat)
#         x_flat = self.upper.solve_flat(w_flat)
#         return x_flat
#
#     def log_det_project(self):
#         return self.upper.log_det_project()
