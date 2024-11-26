from typing import Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.finite.residual.base import ClassicResidualBijection
from torchflows.bijections.matrices import UpperTriangularInvertibleMatrix, HouseholderOrthogonalMatrix, \
    IdentityMatrix, PermutationMatrix
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
        self.register_module('r', UpperTriangularInvertibleMatrix(n_dim=self.m))
        self.register_module('r_tilde', UpperTriangularInvertibleMatrix(n_dim=self.m))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Sylvester bijection does not support forward computation.")

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z_flat = torch.flatten(z, start_dim=len(batch_shape))

        # Prepare parameters
        q = self.q.mat()[:, :self.m]
        r = self.r.mat()
        r_tilde = self.r_tilde.mat()

        u = torch.einsum('...ij,...jk->...ik', q, r)
        u = u.view(*([1] * len(batch_shape)), *u.shape).to(z)

        w = torch.einsum('...ij,...kj->...ik', r_tilde, q)
        w = w.view(*([1] * len(batch_shape)), *w.shape).to(z)

        b = self.b.view(*([1] * len(batch_shape)), *self.b.shape).to(z)

        # Intermediate computations
        wzpb = torch.einsum('...ij,...j->...i', w, z_flat) + b  # (..., m)
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
        self.register_module('q', HouseholderOrthogonalMatrix(n_dim=self.n_dim, n_factors=self.m))


class IdentitySylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.register_module('q', IdentityMatrix(n_dim=self.n_dim))


Sylvester = IdentitySylvester


class PermutationSylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.register_module('q', PermutationMatrix(n_dim=self.n_dim))
