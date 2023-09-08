from typing import Union, Tuple
import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.base import ConditionalBijection
from normalizing_flows.utils import get_batch_shape


class Planar(ConditionalBijection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inv_planar = InversePlanar(*args, **kwargs)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inv_planar.inverse(z=x, context=context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inv_planar.forward(x=z, context=context)


class InversePlanar(ConditionalBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.w = nn.Parameter(torch.randn(size=(self.n_dim,)))
        self.u = nn.Parameter(torch.randn(size=(self.n_dim,)))
        self.b = nn.Parameter(torch.randn(size=()))

    def h(self, x):
        return torch.sigmoid(x)

    def h_deriv(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        u = self.u.view(*([1] * len(batch_shape)), *self.u.shape)
        w = self.w.view(*([1] * len(batch_shape)), *self.w.shape)

        z = z.view(*batch_shape, self.n_dim)
        # x = z + u * self.h(w.T @ z + self.b)
        x = z + u * self.h(torch.einsum('...i,...i', w, z) + self.b)

        # phi = self.h_deriv(w.T @ z + self.b) * w
        phi = self.h_deriv(torch.einsum('...i,...i', w, z) + self.b) * w

        # log_det = torch.log(torch.abs(1 + u.T @ phi))
        log_det = torch.log(torch.abs(1 + torch.einsum('...i,...i', u, phi)))
        x = x.view(*batch_shape, *self.event_shape)

        return x, log_det
