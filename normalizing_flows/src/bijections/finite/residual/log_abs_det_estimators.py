from typing import Any

import torch
import torch.nn as nn

from normalizing_flows.src.utils import Geometric


def hutchinson_log_abs_det_estimator(g: callable, x: torch.Tensor, noise: torch.Tensor, training: bool,
                                     n_iterations: int = 8):
    vjp = noise
    logdetgrad = torch.tensor(0.).to(x)
    dist = Geometric(probs=torch.tensor(0.5), minimum=1)
    for k in range(1, n_iterations + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * noise.view(x.shape[0], -1), 1)
        p_k = 1 - dist.cdf(torch.tensor(k - 1, dtype=torch.long))
        delta = (-1) ** (k + 1) / (k * p_k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_log_abs_det_estimator(g_value: torch.Tensor, x: torch.Tensor, noise: torch.Tensor, training: bool):
    """
    Estimate log[abs(det(grad(f)))](x) with a roulette approach, where f(x) = x + g(x); Lip(g) < 1.

    :param g_value: tensor g(x).
    :param x: input tensor.
    :param noise: noise tensor with the same shape as x.
    :param training: is the computation being performed for a model being trained.
    :return:
    """
    vjp = noise
    neumann_vjp = noise
    dist = Geometric(probs=torch.tensor(0.5), minimum=1)
    n_power_series = int(dist.sample())
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g_value, x, vjp, retain_graph=True)[0]
            # P(N >= k) = 1 - P(N < k) = 1 - P(N <= k - 1) = 1 - cdf(k - 1)
            p_k = 1 - dist.cdf(torch.tensor(k - 1, dtype=torch.long))
            neumann_vjp = neumann_vjp + (-1) ** k / (k * p_k) * vjp
    vjp_jac = torch.autograd.grad(g_value, x, neumann_vjp, create_graph=training)[0]
    return torch.sum(vjp_jac.view(x.shape[0], -1) * noise.view(x.shape[0], -1), 1)


class LogDeterminantEstimator(torch.autograd.Function):
    """
    Given a function f(x) = x + g(x) with Lip(g) < 1, compute log[abs(det(grad(f)))](x) with Pytorch autodiff support.
    Autodiff support permits this function to be used in a computation graph.
    """

    # https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py#L186
    @staticmethod
    def forward(ctx,
                estimator_function: callable,
                g: nn.Module,
                x: torch.Tensor,
                noise: torch.Tensor,
                training: bool,
                *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g_value = g(x)
            ctx.g_value = g_value
            ctx.x = x
            log_abs_det_jacobian = estimator_function(g_value, x, noise, training)

            if training:
                # If a model is being trained,
                # compute the gradient of the log determinant in the forward pass and store it for later.
                # The gradient is computed w.r.t. x (first output) and w.r.t. the parameters of g (following outputs).
                grad_x, *grad_params = torch.autograd.grad(
                    log_abs_det_jacobian.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        # Return g(x) and log(abs(det(grad(f))))(x)
        return (
            g_value.detach().requires_grad_(g_value.requires_grad),
            log_abs_det_jacobian.detach().requires_grad_(log_abs_det_jacobian.requires_grad)
        )

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g_value, x = ctx.g_value, ctx.x

            # Precomputed gradients
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g_value, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from log determinant.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def log_det_roulette(g: nn.Module, x: torch.Tensor, training: bool = False):
    noise = torch.randn_like(x)
    return LogDeterminantEstimator.apply(
        neumann_log_abs_det_estimator,
        g,
        x,
        noise,
        training,
        *list(g.parameters())
    )


def log_det_hutchinson(g: nn.Module, x: torch.Tensor, training: bool = False):
    noise = torch.randn_like(x)
    return LogDeterminantEstimator.apply(
        hutchinson_log_abs_det_estimator,
        g,
        x,
        noise,
        training,
        *list(g.parameters())
    )
