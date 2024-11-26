from typing import Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.finite.matrix.base import InvertibleMatrix


class HouseholderProductMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_factors: int = None, **kwargs):
        super().__init__(event_shape, **kwargs)
        if n_factors is None:
            n_factors = min(5, self.n_dim // 2)
        assert 1 <= n_factors <= self.n_dim

        self.v = nn.Parameter(torch.randn(n_factors, self.n_dim) / self.n_dim ** 2 + torch.eye(n_factors, self.n_dim))
        # self.v = nn.Parameter(torch.randn(n_factors, self.n_dim))
        self.tau = 2

    def apply_flat_transformation(self, x_flat: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        batch_shape = x_flat.shape[:-1]
        z_flat = x_flat.clone()  # (*batch_shape, self.n_dim)
        assert len(factors) == self.v.shape[0]
        for v in factors:
            # v.shape == (self.n_dim,)
            dot = torch.einsum('i,...i->...', v, z_flat)[..., None]  # (*batch_shape, self.n_dim)
            v_unsqueezed = v[[None] * len(batch_shape)]  # (*batch_shape, self.n_dim)
            scalar = self.tau / torch.sum(torch.square(v))
            z_flat = z_flat - scalar * (v_unsqueezed * dot).squeeze(-1)
        return z_flat

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return self.apply_flat_transformation(x_flat, self.v)

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.apply_flat_transformation(b_flat, self.v.flip(0))

    def log_det_project(self) -> torch.Tensor:
        return torch.zeros(1).to(self.device_buffer.device)

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.project_flat(other)
