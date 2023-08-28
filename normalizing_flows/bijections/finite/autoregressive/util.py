from functools import lru_cache
from typing import Any, Tuple, Union, List

import numpy as np
import torch.autograd


# Adapted from https://github.com/francois-rozet/zuko/blob/ee03dba8aa73c62420cbd87359499c0c9aadab63/zuko/utils.py


class Bisection(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, f, y, a: torch.Tensor, b: torch.Tensor, n: int, h: List[torch.Tensor]) -> torch.Tensor:
        ctx.f = f
        ctx.save_for_backward(*h)
        for _ in range(n):
            c = (a + b) / 2
            mask = torch.as_tensor(f(c, h) < y)
            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)
        ctx.x = (a + b) / 2
        return ctx.x

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        f, x = ctx.f, ctx.x
        h = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            y = f(x)
        jac = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)[0]
        grad_y = grad_x / jac
        if h:
            grad_h = torch.autograd.grad(y, h, -grad_y, retain_graph=True)
        else:
            grad_h = ()

        return None, grad_y, None, None, None, *grad_h


def bisection(f, y, a, b, n, h):
    return Bisection.apply(f, y, a.to(y), b.to(y), n, h)


class GaussLegendre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, a: torch.Tensor, b: torch.Tensor, n: int, h: List[torch.Tensor]) -> torch.Tensor:
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *h)
        return GaussLegendre.quadrature(f, a, b, n, h)

    @staticmethod
    def backward(ctx, grad_area: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        f, n = ctx.f, ctx.n
        a, b, *h = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a, h) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b, h) * grad_area
        else:
            grad_b = None

        if h:
            with torch.enable_grad():
                area = GaussLegendre.quadrature(f, a.detach(), b.detach(), n, h)
            grad_h = torch.autograd.grad(area, h, grad_area, retain_graph=True)
        else:
            grad_h = ()

        return (None, grad_a, grad_b, None, *grad_h)

    @staticmethod
    def quadrature(f, a: torch.Tensor, b: torch.Tensor, n: int, h: List[torch.Tensor]) -> torch.Tensor:
        nodes, weights = GaussLegendre.nodes(n, dtype=a.dtype, device=a.device)
        nodes = torch.lerp(
            a[..., None],
            b[..., None],
            nodes,
        ).movedim(-1, 0)
        h_repeated = [h[i].repeat(n, 1, 1) for i in range(len(h))]
        values = f(nodes.reshape(-1, 1, 1), h_repeated).view_as(nodes)
        return (b - a) * torch.tensordot(weights, values, dims=1)

    @staticmethod
    @lru_cache(maxsize=None)
    def nodes(n: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        kwargs.setdefault('dtype', torch.get_default_dtype())
        return torch.as_tensor(nodes, **kwargs), torch.as_tensor(weights, **kwargs)


def gauss_legendre(f, a, b, n, h):
    return GaussLegendre.apply(f, a, b, n, h)
