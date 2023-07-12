import torch


# Note: backpropagating through stochastic trace estimation sounds terrible, but apparently people do it.

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


def log_det_roulette(g: callable, x: torch.Tensor, dist: torch.distributions.Distribution):
    n = dist.sample()
    n_dim = x.shape[-1]
    log_det = 0.0  # TODO batch the log det computation
    v = torch.randn(n_dim, 1)
    for k in range(1, n + 1):
        vjp = torch.autograd.functional.vjp(func=g, inputs=v, create_graph=True) @ v
        log_det += (-1.0) ** (k + 1) / k / dist.icdf(torch.tensor(k, dtype=torch.long)) * vjp
    log_det /= n
    return log_det
