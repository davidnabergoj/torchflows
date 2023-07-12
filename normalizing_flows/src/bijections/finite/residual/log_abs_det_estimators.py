from typing import Any

import torch

from normalizing_flows.src.utils import Geometric


# Note: backpropagating through stochastic trace estimation sounds terrible, but apparently people do it.
# TODO We should manually set the log determinant gradients using the analytic expressions derived for the
#  hutchinson trace estimator and the roulette estimator. This way, we have a stochastic gradient of a deterministic
#  function instead of a deterministic gradient of a stochastic function.

def log_det_hutchinson(g: callable, x: torch.Tensor, n_iterations: int = 8):
    # FIXME log det depends on x, fix it
    n_dim = x.shape[-1]
    vt = torch.randn(n_dim, 1)
    wt = torch.zeros_like(vt) + vt  # Copy of vt
    log_det = 0.0
    for k in range(1, n_iterations + 1):
        # TODO check if create_graph is needed
        wt = torch.autograd.functional.vjp(func=g, inputs=wt.T, create_graph=True)
        log_det += (-1.0) ** (k + 1.0) * wt @ vt.T / k
    return log_det


def log_det_hutchinson_gradient(g: callable, x: torch.Tensor, n_iterations: int = 8):
    raise NotImplementedError


class LogDetHutchinson(torch.autograd.Function):
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


def log_det_roulette(g: callable, x: torch.Tensor, dist: torch.distributions.Distribution = None):
    if dist is None:
        dist = Geometric(probs=torch.tensor(0.5), minimum=1)
    n = dist.sample()
    n_dim = x.shape[-1]
    log_det = 0.0  # TODO batch the log det computation
    v = torch.randn(n_dim, 1)
    for k in range(1, n + 1):
        vjp = torch.autograd.functional.vjp(func=g, inputs=v, create_graph=True) @ v
        log_det += (-1.0) ** (k + 1) / k / dist.icdf(torch.tensor(k, dtype=torch.long)) * vjp
    log_det /= n
    return log_det


def log_det_roulette_gradient(g: callable, x: torch.Tensor, dist: torch.distributions.Distribution = None):
    if dist is None:
        dist = Geometric(probs=torch.tensor(0.5), minimum=1)
    raise NotImplementedError


class LogDetRoulette(torch.autograd.Function):
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
