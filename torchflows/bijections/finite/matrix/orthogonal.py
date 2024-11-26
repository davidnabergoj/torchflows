from typing import Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.finite.matrix.base import InvertibleMatrix


class HouseholderOrthogonalMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_factors: int = None, **kwargs):
        super().__init__(event_shape, **kwargs)
        if n_factors is None:
            n_factors = min(5, self.n_dim)
        assert 1 <= n_factors <= self.n_dim

        self.v = nn.Parameter(torch.randn(n_factors, self.n_dim) / self.n_dim ** 2 + torch.eye(n_factors, self.n_dim))

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        batch_shape = x_flat.shape[:-1]
        z_flat = x_flat.clone()  # (*batch_shape, self.n_dim)
        for i in range(self.v.shape[0]):  # Apply each Householder transformation in reverse order
            v = self.v[i]  # (self.n_dim,)
            alpha = 2 * torch.einsum('i,...i->...', v, z_flat)[..., None]  # (*batch_shape, 1)
            v = v[[None] * len(batch_shape)]  # (1, ..., 1, self.n_dim) with len(v.shape) == len(batch_shape) + 1
            z_flat = z_flat - alpha * v
        return z_flat

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # Same code as project, just the reverse matrix order
        batch_shape = b_flat.shape[:-1]
        x_flat = b_flat.clone()  # (*batch_shape, self.n_dim)
        for i in range(self.v.shape[0] - 1, -1, -1):  # Apply each Householder transformation in reverse order
            v = self.v[i]  # (self.n_dim,)
            alpha = 2 * torch.einsum('i,...i->...', v, x_flat)[..., None]  # (*batch_shape, 1)
            v = v[[None] * len(batch_shape)]  # (1, ..., 1, self.n_dim) with len(v.shape) == len(batch_shape) + 1
            x_flat = x_flat - alpha * v
        return x_flat

    def log_det_project(self) -> torch.Tensor:
        return torch.tensor(0.0).to(self.device_buffer.device)

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.project_flat(other)