from typing import Union, Tuple

import torch
import torch.nn as nn
from Cython.Shadow import returns

from torchflows.bijections.finite.matrix import UpperTriangularInvertibleMatrix, IdentityMatrix, \
    HouseholderOrthogonalMatrix
from torchflows.bijections.finite.matrix.permutation import PermutationMatrix, RandomPermutationMatrix
from torchflows.bijections.finite.matrix.util import matmul_with_householder
from torchflows.bijections.finite.residual.base import ClassicResidualBijection
from torchflows.utils import get_batch_shape


class BaseSylvester(ClassicResidualBijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 m: int = None,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        if m is None:
            m = self.n_dim // 2
        if m > self.n_dim:
            raise ValueError
        self.m = m

        self.register_parameter('b', nn.Parameter(torch.randn(m)))
        self.register_module('r', UpperTriangularInvertibleMatrix((m,)))
        self.register_module('r_tilde', UpperTriangularInvertibleMatrix((m,)))

    def compute_u(self):
        # u = Q * R
        raise NotImplementedError

    def compute_w(self):
        # w = R_tilde * Q.T
        raise NotImplementedError

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Sylvester bijection does not support forward computation.")

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z_flat = torch.flatten(z, start_dim=len(batch_shape))

        # Prepare parameters
        u = self.compute_u()
        w = self.compute_w()

        # Intermediate computations
        wzpb = torch.einsum('ij,...j->...i', w, z_flat) + self.b[[None] * len(batch_shape)]  # (..., m)
        h = torch.sigmoid(wzpb)
        h_deriv = h * (1 - h)
        wu = torch.einsum('...ij,...jk->...ik', w, u)  # (..., m, m)

        # diag = torch.diag(h_deriv)[[None] * len(batch_shape)].repeat(*batch_shape, 1, 1)
        diag = torch.zeros(size=(*batch_shape, self.m, self.m)).to(z)
        diag[..., range(self.m), range(self.m)] = h_deriv  # (..., m, m)

        # Compute the log determinant and output
        _, log_det = torch.linalg.slogdet(torch.eye(self.m).to(z) + torch.einsum('...ij,...jk->...ik', diag, wu))
        x = z_flat + torch.einsum('...ij,...j->...i', u, h)
        x = x.view(*batch_shape, *self.event_shape)

        return x, log_det


class HouseholderSylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.register_module('q', HouseholderOrthogonalMatrix(event_shape, n_factors=self.m))

    def compute_u(self):
        return self.q.project_flat(self.r.compute_matrix())

    def compute_w(self):
        # No need to transpose as Q is symmetric
        return matmul_with_householder(self.r_tilde.compute_matrix(), self.q)


class IdentitySylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.register_module('q', IdentityMatrix(event_shape))

    def compute_u(self):
        r = self.r.compute_matrix()
        return torch.concat([r, torch.zeros_like(r)], dim=-2)

    def compute_w(self):
        rt = self.r_tilde.compute_matrix()
        return torch.concat([rt, torch.zeros_like(rt)], dim=-1)


class PermutationSylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.register_module('q', RandomPermutationMatrix(event_shape))

    def compute_u(self):
        return self.r.compute_matrix()

    def compute_w(self):
        return self.r_tilde.compute_matrix()


# class IdentitySylvester(ClassicResidualBijection):
#     def __init__(self,
#                  event_shape: Union[torch.Size, Tuple[int, ...]],
#                  m: int = None,
#                  **kwargs):
#         super().__init__(event_shape, **kwargs)
#         self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))
#         if m is None:
#             m = self.n_dim // 2
#         if m > self.n_dim:
#             raise ValueError
#         self.m = m
#
#         self.register_parameter('b', nn.Parameter(torch.randn(m)))
#         self.register_module('r', UpperTriangularInvertibleMatrix(event_shape))
#         self.register_module('r_tilde', UpperTriangularInvertibleMatrix(event_shape))
#
#     def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         raise ValueError("Sylvester bijection does not support forward computation.")
#
#     def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_shape = get_batch_shape(z, self.event_shape)
#         z_flat = torch.flatten(z, start_dim=len(batch_shape))
#
#         # Prepare parameters
#         q = torch.eye(self.n_dim, self.m)
#
#         u = torch.einsum('...ij,...jk->...ik', q, self.r)
#         u = u.view(*([1] * len(batch_shape)), *u.shape).to(z)
#         w = torch.concat([self.r_tilde, torch.zeros_like(self.r_tilde)], dim=-1)
#         w = w.view(*([1] * len(batch_shape)), *w.shape).to(z)
#         b = self.b.view(*([1] * len(batch_shape)), *self.b.shape).to(z)
#
#         # Intermediate computations
#         wzpb = torch.einsum('...ij,...j->...i', w, z_flat) + b  # (..., m)
#         h = torch.sigmoid(wzpb)
#         h_deriv = h * (1 - h)
#         wu = torch.einsum('...ij,...jk->...ik', w, u)  # (..., m, m)
#
#         # diag = torch.diag(h_deriv)[[None] * len(batch_shape)].repeat(*batch_shape, 1, 1)
#         diag = torch.zeros(size=(*batch_shape, self.m, self.m)).to(z)
#         diag[..., range(self.m), range(self.m)] = h_deriv  # (..., m, m)
#
#         # Compute the log determinant and output
#         _, log_det = torch.linalg.slogdet(torch.eye(self.m).to(z) + torch.einsum('...ij,...jk->...ik', diag, wu))
#         x = z_flat + torch.einsum('...ij,...j->...i', u, h)
#         x = x.view(*batch_shape, *self.event_shape)
#
#         return x, log_det

Sylvester = IdentitySylvester
