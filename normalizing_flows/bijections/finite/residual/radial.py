from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softplus

from normalizing_flows.bijections.finite.base import Bijection
from normalizing_flows.utils import get_batch_shape


class Radial(Bijection):
    # as per Rezende, Mohamed (2015)

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.beta = nn.Parameter(torch.randn(size=()))
        self.unconstrained_alpha = nn.Parameter(torch.randn(size=()))
        self.z0 = nn.Parameter(torch.randn(size=(self.n_dim,)))

    @property
    def alpha(self):
        return softplus(self.unconstrained_alpha)

    def h(self, z):
        batch_shape = z.shape[:-1]
        z0 = self.z0.view(*([1] * len(batch_shape)), *self.z0.shape)
        r = torch.abs(z - z0)
        return 1 / (self.alpha + r)

    def h_deriv(self, z):
        batch_shape = z.shape[:-1]
        z0 = self.z0.view(*([1] * len(batch_shape)), *self.z0.shape)
        sign = (-1.0) ** torch.where(z - z0 < 0)[0]
        return -(self.h(z) ** 2) * sign * z

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z = z.view(*batch_shape, self.n_dim)
        z0 = self.z0.view(*([1] * len(batch_shape)), *self.z0.shape)

        # Compute transformed point
        x = z + self.beta * self.h(z) * (z - z0)

        # Compute determinant of the Jacobian
        h_val = self.h(z)
        r = torch.abs(z - z0)
        beta_times_h_val = self.beta * h_val
        # det = (1 + self.beta * h_val) ** (self.n_dim - 1) * (1 + self.beta * h_val + self.h_deriv(z) * r)
        # log_det = torch.log(torch.abs(det))
        # log_det = (self.n_dim - 1) * torch.log1p(beta_times_h_val) + torch.log(1 + beta_times_h_val + self.h_deriv(z) * r)
        log_det = torch.abs(torch.add(
            (self.n_dim - 1) * torch.log1p(beta_times_h_val),
            torch.log(1 + beta_times_h_val + self.h_deriv(z) * r)
        ))
        x = x.view(*batch_shape, *self.event_shape)

        return x, log_det
