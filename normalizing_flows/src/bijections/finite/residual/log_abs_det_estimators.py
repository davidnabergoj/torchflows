from typing import Any

import torch

from normalizing_flows.src.utils import Geometric


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


class LogDetHutchinson(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class LogDetRoulette(torch.autograd.Function):
    @staticmethod
    def neumann_logdet_estimator(g, x, vareps, training):
        vjp = vareps
        neumann_vjp = vareps
        dist = Geometric(probs=torch.tensor(0.5), minimum=1)
        n_power_series = int(dist.sample())
        with torch.no_grad():
            for k in range(1, n_power_series + 1):
                vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
                # P(N >= k) = 1 - P(N < k) = 1 - P(N <= k - 1) = 1 - cdf(k - 1)
                p_k = 1 - dist.cdf(torch.tensor(k - 1, dtype=torch.long))
                neumann_vjp = neumann_vjp + (-1) ** k / (k * p_k) * vjp
        vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
        logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        return logdetgrad

    # https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py#L186
    @staticmethod
    def forward(ctx, gnet, x, vareps, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = LogDetRoulette.neumann_logdet_estimator(g, x, vareps, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return (
            g.detach().requires_grad_(g.requires_grad),
            logdetgrad.detach().requires_grad_(logdetgrad.requires_grad)
        )

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def log_det_roulette(g: callable, x: torch.Tensor):
    return LogDetRoulette.apply(x, g)
