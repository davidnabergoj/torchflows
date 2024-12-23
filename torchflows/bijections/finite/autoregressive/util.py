# This file incorporates work covered by the following copyright and permission notice:
#
#   MIT License
#
#   Copyright (c) 2022 The Probabilists
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

# Adapted from https://github.com/francois-rozet/zuko/blob/ee03dba8aa73c62420cbd87359499c0c9aadab63/zuko/utils.py

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch.autograd


class GaussLegendre(torch.autograd.Function):
    """
    Autograd function that computes the Gauss-Legendre quadrature.
    """

    @staticmethod
    def forward(ctx, f, a: torch.Tensor, b: torch.Tensor, n: int, *h: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward autograd map.
        """
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *h)
        return GaussLegendre.quadrature(f, a, b, n, h)

    @staticmethod
    def backward(ctx, grad_area: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Inverse autograd map.
        """
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
    """
    Compute the Gauss-Legendre quadrature.
    """
    return GaussLegendre.apply(f, a, b, n, *h)
