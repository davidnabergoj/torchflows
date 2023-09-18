from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.residual.base import ResidualBijection
from normalizing_flows.bijections.finite.residual.log_abs_det_estimators import log_det_hutchinson
from normalizing_flows.utils import sum_except_batch


# Adapted from: https://github.com/johertrich/Proximal_Residual_Flows/blob/master/prox_res_flow.py

def orthogonal_stiefel_projection(t, n_iterations):
    # Projects matrices T with shape (batch_size, n, d) to the orthogonal Stiefel manifold.
    # Note: probably need to keep n_iterations small because we are backpropagating through this...
    # This is the same as the official implementation, but sounds like we would be better off with a gradient
    # approximation.

    batch_size, n, n_dim = t.shape

    y = t
    for i in range(n_iterations):
        y_transposed = torch.transpose(y, 1, 2)
        batch_identity = torch.eye(n_dim)[None].repeat(batch_size, 1, 1)
        inverse_term = torch.linalg.inv(batch_identity + torch.matmul(y_transposed, y))
        y = 2 * torch.matmul(y, inverse_term)
    return y


class PNNBlock(nn.Module):
    """
    Proximal neural network block.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.b = nn.Parameter(torch.randn(self.hidden_size))
        self.t_tilde = nn.Parameter(torch.randn(self.hidden_size, self.input_size))
        self.alpha = None  # parameters for self.sigma. Set to None at the moment, since we are using TanH.

    @staticmethod
    def sigma(x):
        # sigma is a proximity operator wrt a function g with 0 as a minimizer IFF sigma is 1 lip-cont,
        # monotone increasing and sigma(0) = 0.
        # Tanh qualifies. In fact, many activations do. They are listed in https://arxiv.org/abs/1808.07526v2.
        return torch.tanh(x)

    @staticmethod
    def sigma_prime(x):
        # TanH derivative
        return 4 / torch.square((torch.exp(x) + torch.exp(-x)))

    @property
    def stiefel_matrix(self, n_iterations: int = 4):
        # has shape (n_hidden, n_inputs)
        return orthogonal_stiefel_projection(t=self.t_tilde, n_iterations=n_iterations)

    def regularization(self):
        # to be applied during optimization
        return torch.linalg.norm(self.t_tilde.T @ self.t_tilde - torch.eye(self.input_size))

    def forward(self, x):
        """
        x.shape = (batch_size, input_size)
        """
        mat = self.stiefel_matrix
        act = self.sigma(torch.nn.functional.linear(x, mat, self.b))
        return torch.einsum('...ij,...jk->...ik', mat.T, act)


class ProximalNeuralNetwork(nn.Sequential):
    def __init__(self, input_size: int, n_layers: int, hidden_size: int = 100):
        super().__init__(*[PNNBlock(input_size, hidden_size) for _ in range(n_layers)])
        self.n_layers = n_layers


class ProximalResidualFlowIncrement(nn.Module):
    def __init__(self, pnn: ProximalNeuralNetwork, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.pnn = pnn

    def forward(self, x):
        return self.gamma * self.pnn(x)

    def log_det_single_layer(self, x):
        # Computes the log determinant of the jacobian for a single layer proximal neural network.
        assert len(self.pnn) == 0
        layer = self.pnn[0]
        mat = layer.mat
        b = layer.b

        act_derivative = layer.sigma_prime(torch.nn.functional.linear(x, mat, b))
        log_derivatives = torch.log1p(self.gamma * act_derivative)
        return torch.sum(log_derivatives, dim=-1)


class ProximalResidualFlow(ResidualBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], gamma: float = 1.99, **kwargs):
        super().__init__(event_shape)
        assert gamma > 0
        self.g = ProximalResidualFlowIncrement(
            pnn=ProximalNeuralNetwork(input_size=self.n_dim, **kwargs),
            gamma=gamma
        )

    def log_det(self, x, **kwargs):
        if self.g.pnn.n_layers == 1:
            return self.g.log_det(x)
        else:
            # TODO check
            return log_det_hutchinson(self.g, x, **kwargs)[1]
