from typing import Union, Tuple

import torch
import torch.nn as nn
from normalizing_flows.bijections.continuous.base import ExactODEFunction, TimeDerivative, ExactContinuousBijection


class OTResNet(nn.Module):
    """
    Two-layer ResNet as described in the original OTFlow paper.
    """

    def __init__(self, event_size: int, hidden_size: int, step_size: float = 1.0):
        """
        :param hidden_size: number of hidden dimensions "m".
        :param step_size: "h" in the original formulation.
        """
        super().__init__()

        self.K0 = nn.Parameter(torch.randn(size=(hidden_size, event_size)) / (2 * event_size))
        self.b0 = nn.Parameter(torch.randn(size=(hidden_size,)) / event_size)

        self.K1 = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)) / (2 * event_size))
        self.b1 = nn.Parameter(torch.randn(size=(hidden_size,)) / event_size)

        self.step_size = step_size

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

    def compute_z1(self, s, w, u0: torch.Tensor = None):
        if u0 is None:
            u0 = self.compute_u0(s)
        linear_in = torch.matmul(torch.diag(self.sigma_prime(torch.nn.functional.linear(u0, self.K1, self.b1))), w)
        z1 = w + self.step_size * torch.nn.functional.linear(linear_in, self.K1.T)
        return z1

    def jvp(self, s, w):
        """
        Compute grad_ResNet_wrt_s * w. The first term is the jacobian matrix.
        """
        z1 = self.compute_z1(s, w)
        linear_in_2 = torch.matmul(torch.diag(self.sigma_prime(torch.nn.functional.linear(s, self.K0, self.b0))), z1)
        z0 = torch.nn.functional.linear(linear_in_2, self.K0.T)
        return z0

    def hessian_trace(self,
                      s: torch.Tensor = None,
                      w: torch.Tensor = None,
                      u0: torch.Tensor = None,
                      z1: torch.Tensor = None):
        if u0 is None:
            u0 = self.compute_u0(s)
        if z1 is None:
            z1 = self.compute_z1(s, w, u0=u0)
        # Compute the first term in Equation 14
        t0 = torch.matmul(
            (self.sigma_prime_prime(torch.nn.functional.linear(s, self.K0, self.b0)) * z1).T,
            self.K0[:, :-1] ** 2 @ torch.ones(size=(self.K0.shape[0] - 1,))
        )

        K1J = self.K0.T @ self.sigma_prime(torch.nn.functional.linear(s, self.K0, self.b0))
        t1 = torch.matmul(
            (self.sigma_prime_prime(torch.nn.functional.linear(u0, self.K1, self.b1)) * z1).T,
            K1J ** 2 @ torch.ones(size=(self.K0.shape[0] - 1,))
        )

        return t0 + self.step_size * t1


class OTPotential(TimeDerivative):
    def __init__(self, event_size: int, hidden_size: int, **kwargs):
        super().__init__()
        # hidden_size = m
        r = min(10, event_size)
        self.w = nn.Parameter(torch.randn(size=(hidden_size,)))
        self.A = nn.Parameter(torch.randn(size=(r, event_size)))
        self.b = nn.Parameter(torch.randn(size=(event_size,)))
        self.resnet = OTResNet(event_size, hidden_size, **kwargs)

    def forward(self, t, x):
        # Concatenate t to the end of x
        s = torch.nn.functional.pad(torch.clone(x), pad=(0, 1), value=1.0)
        s[..., -1] = t
        return -self.gradient(s)

    def gradient(self, s):
        # Equation 12
        return self.resnet.jvp(s, self.w) + self.A.T @ self.A @ s + self.b

    def hessian_trace(self, s: torch.Tensor, w: torch.Tensor, u0: torch.Tensor = None, z1: torch.Tensor = None):
        # Equation 14
        tr_first_term = self.resnet.hessian_trace(s, w, u0, z1)

        # E.T @ A ... remove last row (assuming E has d of d+1 standard basis vectors)
        # A @ E ... remove last column (assuming E has d of d+1 standard basis vectors)
        tr_second_term = torch.trace((self.A.T @ self.A)[:-1, :-1])

        return tr_first_term + tr_second_term


class OTFlow(ExactContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = ExactODEFunction(OTPotential(n_dim, hidden_size=30))
        super().__init__(event_shape, diff_eq, **kwargs)
