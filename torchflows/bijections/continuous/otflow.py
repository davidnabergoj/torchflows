import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchflows.bijections.continuous.base import ExactODEFunction, TimeDerivative, ExactContinuousBijection


def concatenate_x_t(x, t):
    # Concatenate t to the end of x
    s = torch.nn.functional.pad(torch.clone(x), pad=(0, 1), value=1.0)
    s[..., -1] = t
    return s


class OTResNet(nn.Module):
    """
    Two-layer ResNet as described in the original OTFlow paper.
    """

    def __init__(self, event_size: int, hidden_size: int, step_size: float = 0.01):
        """
        :param hidden_size: number of hidden dimensions "m".
        :param step_size: "h" in the original formulation.
        """
        super().__init__()

        # Initialize K0, K1 close to identity
        # Initialize b0, b1 close to zero

        divisor = max(event_size ** 2, 10)

        self.K0_delta = nn.Parameter(torch.randn(size=(hidden_size, event_size)) / divisor)
        self.b0 = nn.Parameter(torch.randn(size=(hidden_size,)) / divisor)

        self.K1_delta = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)) / divisor)
        self.b1 = nn.Parameter(torch.randn(size=(hidden_size,)) / divisor)

        self.step_size = step_size

    @property
    def K0(self):
        return torch.eye(*self.K0_delta.shape).to(self.K0_delta) + self.K0_delta / 1000

    @property
    def K1(self):
        return torch.eye(*self.K1_delta.shape).to(self.K1_delta) + self.K1_delta / 1000

    @staticmethod
    def sigma(x):
        return torch.log(torch.exp(x) + torch.exp(-x))

    @staticmethod
    def sigma_prime(x):
        return torch.tanh(x)

    @staticmethod
    def sigma_prime_prime(x):
        # Square of the hyperbolic secant
        return 4 / torch.square(torch.exp(x) + torch.exp(-x))

    def compute_u0(self, s):
        u0 = self.sigma(torch.nn.functional.linear(s, self.K0, self.b0))
        return u0

    def forward(self, s):
        """
        x.shape = (batch_size, event_size)
        """
        u0 = self.compute_u0(s)
        u1 = u0 + self.step_size * self.sigma(torch.nn.functional.linear(u0, self.K1, self.b1))
        return u1

    def compute_z1(self, s, w: torch.Tensor, u0: torch.Tensor = None):
        if u0 is None:
            u0 = self.compute_u0(s)
        linear_in = self.sigma_prime(torch.nn.functional.linear(u0, self.K1, self.b1)) * w[None]
        z1 = w[None] + self.step_size * torch.nn.functional.linear(linear_in, self.K1.T)
        return z1

    def jvp(self, s, w):
        """
        Compute grad_ResNet_wrt_s * w. The first term is the jacobian matrix.
        """
        z1 = self.compute_z1(s, w)
        linear_in_2 = self.sigma_prime(torch.nn.functional.linear(s, self.K0, self.b0)) * z1
        z0 = torch.nn.functional.linear(linear_in_2, self.K0.T)
        return z0

    def hessian_trace(self,
                      s: torch.Tensor,
                      w: torch.Tensor,
                      u0: torch.Tensor = None,
                      z1: torch.Tensor = None):
        if u0 is None:
            u0 = self.compute_u0(s)
        if z1 is None:
            z1 = self.compute_z1(s, w, u0=u0)

        # print(torch.linalg.norm(u0))

        assert torch.all(torch.isfinite(s))
        assert torch.all(~torch.isnan(s))

        assert torch.all(torch.isfinite(w))
        assert torch.all(~torch.isnan(w))

        assert torch.all(torch.isfinite(u0))
        assert torch.all(~torch.isnan(u0))

        assert torch.all(torch.isfinite(z1))
        assert torch.all(~torch.isnan(z1))

        # Compute the first term in Equation 14

        ones = torch.ones(size=(self.K0.shape[1] - 1,)).to(s)

        t0 = torch.sum(
            torch.multiply(
                self.sigma_prime_prime(torch.nn.functional.linear(s, self.K0, self.b0)) * z1,
                torch.nn.functional.linear(ones, self.K0[:, :-1] ** 2)
            ),
            dim=1
        )

        # K1J = self.K1 @ self.K0.T @ self.sigma_prime(torch.nn.functional.linear(s, self.K0, self.b0))
        K1J = torch.matmul(
            self.K1,
            self.sigma_prime(torch.nn.functional.linear(s, self.K0, self.b0))[..., None] * self.K0[:, :-1][None]
        )

        t1 = torch.sum(
            torch.multiply(
                self.sigma_prime_prime(torch.nn.functional.linear(u0, self.K1, self.b1)) * w,
                (K1J ** 2) @ ones
            ),
            dim=1
        )

        return t0 + self.step_size * t1


class OTPotential(TimeDerivative):
    def __init__(self, event_size: int, hidden_size: int = None, **kwargs):
        super().__init__()

        # hidden_size = m
        if hidden_size is None:
            hidden_size = max(int(math.log(event_size)), 4)

        r = min(10, event_size)

        # Initialize w to 1
        # Initialize A to identity
        # Initialize b to 0

        divisor = event_size ** 2

        delta_w = torch.randn(size=(hidden_size,)) / divisor
        delta_A = torch.randn(size=(r, event_size + 1)) / divisor
        delta_b = torch.randn(size=(event_size + 1,)) / divisor

        self.w = nn.Parameter(1 + delta_w)
        self.A = nn.Parameter(torch.eye(r, event_size + 1) + delta_A)
        self.b = nn.Parameter(0 + delta_b)
        self.resnet = OTResNet(event_size + 1, hidden_size, **kwargs)  # (x, t) has d+1 elements

    def forward(self, t, x):
        return self.gradient(concatenate_x_t(x, t))

    def gradient(self, s):
        # Equation 12
        out = self.resnet.jvp(s, self.w) + torch.nn.functional.linear(s, self.A.T @ self.A, self.b)
        return out[..., : -1]  # Remove the time prediction

    def hessian_trace(self, s: torch.Tensor, u0: torch.Tensor = None, z1: torch.Tensor = None):
        assert torch.all(torch.isfinite(s))
        assert torch.all(~torch.isnan(s))

        # Equation 14
        tr_first_term = self.resnet.hessian_trace(s, self.w, u0, z1)

        assert torch.all(torch.isfinite(tr_first_term))
        assert torch.all(~torch.isnan(tr_first_term))

        # E.T @ A ... remove last row (assuming E has d of d+1 standard basis vectors)
        # A @ E ... remove last column (assuming E has d of d+1 standard basis vectors)
        tr_second_term = torch.trace((self.A.T @ self.A)[:-1, :-1])

        assert torch.all(torch.isfinite(tr_second_term))
        assert torch.all(~torch.isnan(tr_second_term))

        return tr_first_term + tr_second_term


class OTFlowODEFunction(ExactODEFunction):
    def __init__(self, n_dim, **kwargs):
        super().__init__(OTPotential(n_dim, **kwargs))

    def compute_log_det(self, t, x):
        return self.diffeq.hessian_trace(concatenate_x_t(x, t)).view(-1, 1)  # Need an empty dim at the end


class OTFlow(ExactContinuousBijection):
    """
    Optimal transport flow (OT-flow) architecture.

    Reference: Onken et al. "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport" (2021); https://arxiv.org/abs/2006.00104.
    """
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = OTFlowODEFunction(n_dim)
        super().__init__(event_shape, diff_eq, **kwargs)
