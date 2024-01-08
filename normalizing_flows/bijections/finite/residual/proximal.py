import math
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.residual.base import ResidualBijection
from normalizing_flows.bijections.finite.residual.log_abs_det_estimators import log_det_roulette


# Adapted from: https://github.com/johertrich/Proximal_Residual_Flows/blob/master/prox_res_flow.py


class ProximityOperator(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor]):
        super().__init__()
        self.alpha = alpha

    @property
    def t(self):
        # Assuming the activation comes from the referenced list
        return 0.5  # Unused

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class TanH(ProximityOperator):
    def __init__(self):
        super().__init__(alpha=None)

    def forward(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 4 / torch.square(torch.exp(x) + torch.exp(-x))


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

    def __init__(self, event_size: int, hidden_size: int, act: ProximityOperator):
        super().__init__()
        self.event_size = event_size
        self.hidden_size = hidden_size

        # Initialize b close to 0
        # Initialize t_tilde close to identity

        divisor = max(self.event_size ** 2, 100)
        self.b = nn.Parameter(torch.randn(self.hidden_size) / divisor)
        self.delta_t_tilde = nn.Parameter(torch.randn(self.hidden_size, self.event_size) / divisor)
        self.act = act

    @property
    def t_tilde(self):
        return torch.eye(self.hidden_size, self.event_size) + self.delta_t_tilde

    @property
    def stiefel_matrix(self, n_iterations: int = 4):
        # output has shape (hidden_size, event_size)
        return orthogonal_stiefel_projection(t=self.t_tilde[None], n_iterations=n_iterations)[0]

    def regularization(self):
        # to be applied during optimization
        # compute T.transpose @ T along the smaller dimension
        if self.hidden_size > self.event_size:
            return torch.linalg.norm(self.t_tilde.T @ self.t_tilde - torch.eye(self.event_size))
        else:
            return torch.linalg.norm(self.t_tilde @ self.t_tilde.T - torch.eye(self.hidden_size))

    def forward(self, x):
        """
        x.shape = (batch_size, event_size)
        """
        mat = self.stiefel_matrix
        act = self.act(torch.nn.functional.linear(x, mat, self.b))
        return torch.einsum('...ij,...kj->...ki', mat.T, act)


class PNN(nn.Sequential):
    """
    Proximal neural network
    """

    def __init__(self, event_size: int, n_layers: int = 1, hidden_size: int = None, act: ProximityOperator = None):
        if act is None:
            act = TanH()
        if hidden_size is None:
            hidden_size = max(math.log(event_size), 4)
        super().__init__(*[PNNBlock(event_size, hidden_size, act) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.act = act

    @property
    def t(self):
        return self.n_layers / (self.n_layers + 1)


class ProximalResFlowBlockIncrement(nn.Module):
    def __init__(self, pnn: PNN, gamma: float, max_gamma: float):
        super().__init__()
        self.gamma = gamma
        self.max_gamma = max_gamma
        assert 0 < gamma < self.max_gamma, f'{gamma = }, {self.max_gamma = }'
        self.phi = pnn

    def r(self, x):
        return 1 / self.phi.t * (self.phi(x) - (1 - self.phi.t) * x)

    def forward(self, x):
        const = self.gamma * self.phi.t / (1 + self.gamma - self.gamma * self.phi.t)
        return const * self.r(x)

    def log_det_single_layer(self, x):
        # Computes the log determinant of the jacobian for a single layer proximal neural network.
        assert self.phi.n_layers == 1
        layer: PNNBlock = self.phi[0]
        mat = layer.stiefel_matrix
        b = layer.b

        act_derivative = layer.act.derivative(torch.nn.functional.linear(x, mat, b))
        log_derivatives = torch.log1p(self.gamma * act_derivative)
        return torch.sum(log_derivatives, dim=-1)


class ProximalResFlowBlock(ResidualBijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 gamma: float = None,
                 n_layers: int = 1,
                 **kwargs):
        # Check: setting low gamma means doing basically nothing to the input. Find a reasonable setting which is still
        # numerically stable.
        super().__init__(event_shape)

        # Set gamma
        assert n_layers > 0

        if n_layers > 1:
            self.max_gamma = (n_layers + 1) / (n_layers - 1)
        else:
            self.max_gamma = 1.5

        if gamma is None:
            gamma = self.max_gamma - 1e-2
        assert 0 < gamma < self.max_gamma

        self.g = ProximalResFlowBlockIncrement(
            pnn=PNN(event_size=self.n_dim, n_layers=n_layers, **kwargs),
            gamma=gamma,
            max_gamma=self.max_gamma
        )

    def log_det(self, x, **kwargs):
        if self.g.phi.n_layers == 1:
            return self.g.log_det_single_layer(x)
        else:
            return log_det_roulette(self.g, x, **kwargs)[1]

    def inverse(self,
                z: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False,
                n_iterations: int = 25,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.g.gamma
        t = self.g.phi.t
        x = z
        for _ in range(n_iterations):
            x = 1 / (1 + gamma - gamma * t) * (z - gamma * t * self.g.r(x))
        if skip_log_det:
            return x
        else:
            log_det = self.log_det(x, **kwargs)
            return x, log_det
