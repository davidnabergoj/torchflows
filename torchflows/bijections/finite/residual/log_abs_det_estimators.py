# This file incorporates work covered by the following copyright and permission notice:
#
#   MIT License
#
#   Copyright (c) 2019 Ricky Tian Qi Chen
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from typing import Tuple, Union

import torch
import torch.nn as nn

from torchflows.utils import Geometric, flatten_batch, get_batch_shape


def power_series_log_abs_det_estimator(g: callable,
                                       x: torch.Tensor,
                                       noise: torch.Tensor,
                                       training: bool,
                                       n_iterations: int = 8):
    # f(x) = x + g(x)
    # x.shape == (batch_size, *event_shape)
    # noise.shape == (batch_size, *event_shape, n_hutchinson_samples)
    # g(x).shape == (batch_size, *event_shape)

    batch_size, *event_shape, n_hutchinson_samples = noise.shape
    event_dims = tuple(range(1, len(x.shape)))
    assert x.shape == (batch_size, *event_shape)
    assert n_iterations >= 2

    w = noise  # (batch_size, event_size, n_hutchinson_samples)
    log_abs_det_jac_f = torch.zeros(
        size=(batch_size,), 
        device=x.device, 
        dtype=x.dtype
    )
    g_value = None

    for k in range(1, n_iterations + 1):
        # Reshape tensors for hutchinson averaging
        x_rep = x[..., None].repeat(
            *([1] * (len(event_shape) + 1)), 
            n_hutchinson_samples
        ).view(
            batch_size * n_hutchinson_samples, 
            *event_shape
        )
        w_flat = w.view(batch_size * n_hutchinson_samples, *event_shape)

        # Compute VJP
        if training:
            gs_r, ws_r = torch.autograd.functional.vjp(
                g, x_rep, w_flat, create_graph=True
            )
        else:
            with torch.no_grad():
                gs_r, ws_r = torch.autograd.functional.vjp(
                    g, x_rep, w_flat, create_graph=False
                )

        if g_value is None:
            g_value = gs_r.view(
                batch_size, 
                *event_shape,
                n_hutchinson_samples
            )[..., 0]

        w = ws_r.view(batch_size, *event_shape, n_hutchinson_samples)

        # sum over event dims, average over hutchinson dim
        log_abs_det_jac_f += (
            ((-1) ** (k + 1)) / k
            * torch.sum(w * noise, dim=event_dims).mean(dim=1)
        )
        assert log_abs_det_jac_f.shape == (batch_size,)

    return g_value, log_abs_det_jac_f


def roulette_log_abs_det_estimator(g, 
                                   x, 
                                   noise, 
                                   training: bool, 
                                   p: float = 0.5):
    """
    Estimate log[abs(det(grad(f)))](x) with a roulette approach, where f(x) = x + g(x); Lip(g) < 1.

    :param g:.
    :param x: input tensor.
    :param noise: noise tensor with the same shape as x.
    :param training: is the computation being performed for a model being trained.
    :return:
    """
    event_dims = tuple(range(1, x.dim()))
    w = noise
    neumann_vjp = noise

    dist = Geometric(
        probs=torch.tensor(
            p, 
            device=x.device, 
            dtype=x.dtype
        ), 
        minimum=1
    )
    n_power_series = int(dist.sample().item())

    # Build or skip graphs depending on training
    def vjp(v, cg): return torch.autograd.functional.vjp(
        g, 
        x, 
        v, 
        create_graph=cg
    )[1]

    if training:
        for k in range(1, n_power_series + 1):
            w = vjp(w, True)
            p_k = 1 - dist.cdf(
                torch.tensor(
                    k - 1,
                    device=x.device, 
                    dtype=torch.long
                )
            )
            neumann_vjp = neumann_vjp + ((-1) ** k) / (k * p_k) * w
    else:
        with torch.no_grad():
            for k in range(1, n_power_series + 1):
                w = vjp(w, False)
                p_k = 1 - dist.cdf(
                    torch.tensor(
                        k - 1,
                        device=x.device, 
                        dtype=torch.long
                    )
                )
                neumann_vjp = neumann_vjp + ((-1) ** k) / (k * p_k) * w

    g_value, vjp_jac = torch.autograd.functional.vjp(
        g, 
        x, 
        neumann_vjp, 
        create_graph=training
    )

    log_abs_det_jac_f = torch.sum(vjp_jac * noise, dim=event_dims)
    return g_value, log_abs_det_jac_f


class LogDeterminantEstimator(torch.autograd.Function):
    """
    Given a function f(x) = x + g(x) with Lip(g) < 1, compute log[abs(det(grad(f)))](x) with Pytorch autodiff support.
    Autodiff support permits this function to be used in a computation graph.
    """

    @staticmethod
    def forward(ctx,
                estimator_function: callable,
                g: torch.Tensor,
                x: torch.Tensor,
                noise: torch.Tensor,
                training: bool,
                *g_params: torch.Tensor):
        # Save what we need to recompute in backward
        ctx.training = training
        ctx.estimator_function = estimator_function
        ctx.g = g
        ctx.save_for_backward(x, noise, *g_params)

        # Forward values without gradient pre-computation
        with torch.set_grad_enabled(training):
            g_value, log_det_f = estimator_function(g, x, noise, training)

        return (
            g_value.detach().requires_grad_(g_value.requires_grad),
            log_det_f.detach().requires_grad_(log_det_f.requires_grad),
        )

    @staticmethod
    def backward(ctx, grad_g, grad_logdet):
        if not ctx.training:
            raise ValueError("Provide training=True if using backward.")

        estimator_function = ctx.estimator_function
        g = ctx.g
        tensors = ctx.saved_tensors
        x, noise = tensors[0], tensors[1]
        g_params = list(tensors[2:])

        # Gradients from g(x, params)
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            for p in g_params:
                if p is not None:
                    p.requires_grad_(True)
            g_value = g(x_)
            dg_x, *dg_params = torch.autograd.grad(
                g_value, [x_] + g_params,
                grad_g, 
                allow_unused=True
            )

        # Gradients from the log determinant
        with torch.enable_grad():
            x2 = x.detach().requires_grad_(True)
            _, log_det_f = estimator_function(g, x2, noise, training=True)
            gl_x, *gl_params = torch.autograd.grad(
                log_det_f, (x2,) + tuple(g_params),
                grad_outputs=grad_logdet, allow_unused=True
            )

        # Combine
        def add(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        grad_x = add(dg_x, gl_x)
        grad_params = [add(a, b) for a, b in zip(dg_params, gl_params)]

        return (None, None, grad_x, None, None) + tuple(grad_params)


def log_det_roulette(event_shape: Union[torch.Size, Tuple[int, ...]],
                     g: nn.Module,
                     x: torch.Tensor,
                     training: bool = False,
                     p: float = 0.5):
    batch_shape = get_batch_shape(x, event_shape)
    x = flatten_batch(x, batch_shape)
    noise = torch.randn_like(x).to(x)
    return LogDeterminantEstimator.apply(
        lambda *args, **kwargs: roulette_log_abs_det_estimator(
            *args, 
            **kwargs, 
            p=p
        ),
        g,
        x,
        noise,
        training,
        *list(g.parameters())
    )


def log_det_power_series(event_shape: Union[torch.Size, Tuple[int, ...]],
                         g: nn.Module,
                         x: torch.Tensor,
                         training: bool = False,
                         n_iterations: int = 8,
                         n_hutchinson_samples: int = 1):
    batch_shape = get_batch_shape(x, event_shape)
    x = flatten_batch(x, batch_shape)
    noise = torch.randn(size=(*x.shape, n_hutchinson_samples)).to(x)
    return LogDeterminantEstimator.apply(
        lambda *args, **kwargs: power_series_log_abs_det_estimator(
            *args, 
            **kwargs, 
            n_iterations=n_iterations
        ),
        g,
        x,
        noise,
        training,
        *list(g.parameters())
    )
