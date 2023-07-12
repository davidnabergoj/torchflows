from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.src.bijections import Bijection
from normalizing_flows.src.bijections.matrices import UpperTriangularInvertibleMatrix, HouseholderOrthogonalMatrix, \
    IdentityMatrix, PermutationMatrix
from normalizing_flows.src.utils import get_batch_shape


class BaseSylvester(Bijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 m: int):
        super().__init__(event_shape)
        self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        self.m = m
        self.b = nn.Parameter(torch.randn(m))

        # q is implemented in subclasses
        self.r = UpperTriangularInvertibleMatrix(n_dim=self.m)
        self.r_tilde = UpperTriangularInvertibleMatrix(n_dim=self.m)

    @property
    def w(self):
        r_tilde = self.r_tilde.mat()
        q = self.q.mat()
        return torch.einsum('...ij,...kj->...ik', r_tilde, q)

    @property
    def u(self):
        r = self.r.mat()
        q = self.q.mat()
        return torch.einsum('...ij,...jk->...ik', q, r)

    def h(self, x):
        return torch.sigmoid(x)

    def h_deriv(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        u = self.u.view(*([1] * len(batch_shape)), *self.u.shape)
        w = self.w.view(*([1] * len(batch_shape)), *self.w.shape)
        b = self.b.view(*([1] * len(batch_shape)), *self.b.shape)

        wzpb = torch.einsum('...ij,...j->...i', w, z) + b  # (..., m)

        z = z.view(*batch_shape, self.n_dim)
        x = z + torch.einsum(
            '...ij,...j->...i',
            u,
            self.h(wzpb)
        )

        wu = torch.einsum('...ij,...jk->...ik', w, u)  # (..., m, m)
        diag = torch.zeros(size=(batch_shape, self.m, self.m))
        diag[..., range(self.m), range(self.m)] = self.h_deriv(wzpb)  # (..., m, m)
        _, log_det = torch.linalg.slogdet(torch.eye(self.m) + torch.einsum('...ij,...jk->...ik', diag, wu))

        x = x.view(*batch_shape, *self.event_shape)

        return x, log_det


class HouseholderSylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], m: int):
        self.q = HouseholderOrthogonalMatrix(n_dim=self.n_dim, n_factors=self.m)
        super().__init__(event_shape, m)


class IdentitySylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], m: int):
        self.q = IdentityMatrix(n_dim=self.n_dim)
        super().__init__(event_shape, m)


class PermutationSylvester(BaseSylvester):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], m: int):
        self.q = PermutationMatrix(n_dim=self.n_dim)
        super().__init__(event_shape, m)
