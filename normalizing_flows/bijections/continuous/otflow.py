from typing import Union, Tuple

import torch
import torch.nn as nn
from normalizing_flows.bijections.continuous.base import ContinuousBijection, ODEFunctionBase, ODEFunction, \
    DifferentialEquationNeuralNetwork


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

    def compute_u0(self, s):
        u0 = torch.nn.functional.linear(s, self.K0, self.b0)
        return u0

    def forward(self, s):
        """
        x.shape = (batch_size, event_size)
        TODO flatten the inputs as they come in or require that they come in flatten. Choose one.
        """
        u0 = self.compute_u0(s)
        u1 = u0 + self.step_size * self.sigma(torch.nn.functional.linear(u0, self.K1, self.b1))
        return u1

    def jvp(self, s, w):
        """
        Compute grad_ResNet_wrt_s * w. The first term is the jacobian matrix.
        """
        u0 = self.compute_u0(s)
        linear_in = torch.matmul(torch.diag(torch.tanh(torch.nn.functional.linear(u0, self.K1, self.b1))), w)
        z1 = w + self.step_size * torch.nn.functional.linear(linear_in, self.K1.T)

        linear_in_2 = torch.matmul(torch.diag(torch.tanh(torch.nn.functional.linear(s, self.K0, self.b0))), z1)
        z0 = torch.nn.functional.linear(linear_in_2, self.K0.T)
        return z0


class OTNetwork(DifferentialEquationNeuralNetwork):
    # TODO introduce a new superclass which OTNetwork and DifferentialEquationNeuralNetwork both inherit.
    def __init__(self, event_size: int, hidden_size: int, **kwargs):
        # hidden_size = m
        super().__init__([])
        r = min(10, event_size)
        self.w = nn.Parameter(torch.randn(size=(hidden_size,)))
        self.A = nn.Parameter(torch.randn(size=(r, event_size)))
        self.b = nn.Parameter(torch.randn(size=(event_size,)))
        self.resnet = OTResNet(event_size, hidden_size, **kwargs)

    def forward(self, t, x):
        s = ...  # TODO concatentate x and t. x goes first, t is at the end.
        grad = self.resnet.jvp(s, self.w) + self.A.T @ self.A @ s + self.b
        return -grad


class OTFlow(ContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], f: ODEFunctionBase, **kwargs):
        super().__init__(event_shape, f, **kwargs)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = ODEFunction(create_nn(n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
