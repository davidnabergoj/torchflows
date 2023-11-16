import math
from typing import Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.utils import get_batch_shape, sum_except_batch


class Affine(Transformer):
    """
    Affine transformer.

    Computes z = alpha * x + beta, where alpha > 0 and -inf < beta < inf.
    Alpha and beta have the same shape as x, i.e. the computation is performed elementwise.
    We use a minimum permitted scale m, 0 < m <= alpha, for numerical stability
    """

    def __init__(self, event_shape: torch.Size, min_scale: float = 1e-3):
        super().__init__(event_shape=event_shape)
        self.m = min_scale
        self.identity_unconstrained_alpha = math.log(1 - self.m)
        self.const = 2

    @property
    def n_parameters(self) -> int:
        return 2 * self.n_dim

    def unflatten_conditioner_parameters(self, h: torch.Tensor):
        return torch.unflatten(h, dim=-1, sizes=(*self.event_shape, 2))

    def forward_base(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.identity_unconstrained_alpha + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        u_beta = h[..., 1]
        beta = u_beta

        log_det = sum_except_batch(log_alpha, self.event_shape)
        return alpha * x + beta, log_det

    def inverse_base(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.identity_unconstrained_alpha + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        u_beta = h[..., 1]
        beta = u_beta

        log_det = -sum_except_batch(log_alpha, self.event_shape)
        return (z - beta) / alpha, log_det


class Shift(Transformer):
    def __init__(self, event_shape: torch.Size):
        super().__init__(event_shape=event_shape)

    @property
    def n_parameters(self) -> int:
        return 1 * self.n_dim

    def unflatten_conditioner_parameters(self, h: torch.Tensor):
        return torch.unflatten(h, dim=-1, sizes=(*self.event_shape, 1))

    def forward_base(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(batch_shape, device=x.device)
        return x + beta, log_det

    def inverse_base(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(batch_shape, device=z.device)
        return z - beta, log_det


class Scale(Transformer):
    """
    Scaling transformer.

    Computes z = alpha * x, where alpha > 0.
    We use a minimum permitted scale m, 0 < m <= alpha, for numerical stability
    """

    def __init__(self, event_shape: torch.Size, min_scale: float = 1e-3):
        super().__init__(event_shape=event_shape)
        self.m = min_scale
        self.const = 2.0
        self.u_alpha_1 = math.log(1 - self.m)

    @property
    def n_parameters(self) -> int:
        return 1 * self.n_dim

    def unflatten_conditioner_parameters(self, h: torch.Tensor):
        return torch.unflatten(h, dim=-1, sizes=(*self.event_shape, 1))

    def forward_base(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.u_alpha_1 + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        log_det = sum_except_batch(log_alpha, self.event_shape)
        return alpha * x, log_det

    def inverse_base(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.u_alpha_1 + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        log_det = -sum_except_batch(log_alpha, self.event_shape)
        return z / alpha, log_det
