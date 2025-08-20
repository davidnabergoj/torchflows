import torch
from typing import Any, Tuple, Union, List


class Bisection(torch.autograd.Function):
    """
    Autograd bisection function.
    """
    @staticmethod
    def forward(ctx: Any, f, y, a: torch.Tensor, b: torch.Tensor, n: int, h: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward method for the autograd function.
        """
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
        """
        Backward method for the autograd function.
        """
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
    """
    Apply bisection with autograd support.
    """
    return Bisection.apply(f, y, a.to(y), b.to(y), n, h)


def bisection_no_gradient(f: callable,
                          y: torch.Tensor,
                          a: Union[torch.Tensor, float] = None,
                          b: Union[torch.Tensor, float] = None,
                          n_iterations: int = 500,
                          atol: float = 1e-9):
    """
    Apply bisection without autograd support.

    Explanation: find x that satisfies f(x) = y. We assume x.shape == y.shape. f is applied elementwise.

    :param f: function that takes as input a tensor and produces as output z.
    :param y: value to match to z.
    :param a: lower bound for bisection search.
    :param b: upper bound for bisection search.
    :param n_iterations: number of bisection iterations.
    :param atol: absolute tolerance.
    """

    if a is None:
        a = torch.full_like(y, fill_value=-100.0)
    elif isinstance(a, float):
        a = torch.full_like(y, fill_value=a)

    if b is None:
        b = torch.full_like(y, fill_value=100.0)
    elif isinstance(b, float):
        b = torch.full_like(y, fill_value=b)

    c = (a + b) / 2
    log_det = None
    for _ in range(n_iterations):
        y_prime, log_det = f(c)
        mask = torch.as_tensor(y_prime < y)
        a = torch.where(mask, c, a)
        b = torch.where(mask, b, c)

        # Check for convergence
        if torch.allclose(c, c := ((a + b) / 2), atol=atol):
            break
    x = c
    return x, -log_det
