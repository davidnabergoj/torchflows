from functools import lru_cache
from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.src.bijections import Bijection
from normalizing_flows.src.utils import get_batch_shape, Geometric


class SpectralLinear(nn.Module):
    # https://arxiv.org/pdf/1811.00995.pdf

    def __init__(self, n_inputs: int, n_outputs: int, c: float = 0.99):
        super().__init__()
        self.c = c
        self.n_inputs = n_inputs
        self.w = torch.randn(n_outputs, n_inputs)
        self.bias = nn.Parameter(torch.randn(n_outputs))

    @lru_cache(maxsize=2)  # TODO check that this works. Meant to avoid repeats without descent.
    def power_iteration(self, w, n_iterations: int = 25):
        # https://arxiv.org/pdf/1804.04368.pdf
        # Get maximum singular value of rectangular matrix w
        x = torch.randn(self.n_inputs, 1)
        for _ in range(n_iterations):
            x = w.T @ w @ x
        return torch.linalg.norm(w @ x) / torch.linalg.norm(x)

    @property
    def normalized_mat(self):
        # Estimate sigma
        sigma = self.power_iteration(self.w)
        ratio = self.c / sigma
        return self.w * (ratio ** (ratio < 1))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.normalized_mat, self.bias)


class InvertibleResNet(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_hidden: int = 100, n_hidden_layers: int = 2):
        super().__init__(event_shape)

        layers = []
        if n_hidden_layers == 0:
            layers = [SpectralLinear(self.n_dim, self.n_dim)]
        else:
            layers.append(SpectralLinear(self.n_dim, n_hidden))
            for _ in range(n_hidden):
                layers.append(nn.Tanh())
                layers.append(SpectralLinear(n_hidden, n_hidden))
            layers.pop(-1)
            layers.append(SpectralLinear(n_hidden, self.n_dim))

        self.g = nn.Sequential(*layers)

    def log_det(self, x: torch.Tensor, n_iterations: int = 8):
        # FIXME log det depends on x, fix it
        vt = torch.randn(self.n_dim, 1)
        wt = torch.zeros_like(vt) + vt  # Copy of vt
        log_det = 0.0
        for k in range(1, n_iterations + 1):
            # TODO check if create_graph is needed
            wt = torch.autograd.functional.vjp(func=self.g.forward, inputs=wt.T, create_graph=True)
            log_det += (-1.0) ** (k + 1.0) * wt @ vt.T / k
        return log_det

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        x = x.view(*batch_shape, self.n_dim)
        z = x + self.g(x)
        z = z.view(*batch_shape)
        log_det = torch.full(size=batch_shape, fill_value=torch.nan if skip_log_det else self.log_det())
        return z, log_det

    def inverse(self,
                z: torch.Tensor,
                context: torch.Tensor = None,
                n_iterations: int = 25,
                skip_log_det: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z = z.view(*batch_shape, self.n_dim)
        x = z
        for _ in range(n_iterations):
            x = z - self.g(x)
        x = x.view(*batch_shape)
        log_det = -torch.full(size=batch_shape, fill_value=torch.nan if skip_log_det else self.log_det())
        return x, log_det


class ResFlow(InvertibleResNet):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], p: float = 0.5):
        super().__init__(event_shape)
        self.dist = Geometric(probs=torch.tensor(p), minimum=1)

    def log_det(self, x: torch.Tensor, **kwargs):
        # kwargs are ignored
        n = self.dist.sample()
        log_det = 0.0  # TODO batch the log det computation
        v = torch.randn(self.n_dim, 1)
        for k in range(1, n + 1):
            vjp = torch.autograd.functional.vjp(func=self.g.forward, inputs=v, create_graph=True) @ v
            log_det += (-1.0) ** (k + 1) / k / self.dist.icdf(torch.tensor(k, dtype=torch.long)) * vjp
        log_det /= n
        return log_det


class QuasiAutoregressiveFlow(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)


class ProximalResFlow(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
