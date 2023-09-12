from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.finite.residual.base import ResidualBijection
from normalizing_flows.bijections.finite.residual.log_abs_det_estimators import log_det_hutchinson, log_det_roulette


class SpectralLinear(nn.Module):
    # https://arxiv.org/pdf/1811.00995.pdf

    def __init__(self, n_inputs: int, n_outputs: int, c: float = 0.97, n_iterations: int = 25):
        super().__init__()
        self.c = c
        self.n_inputs = n_inputs
        self.w = torch.randn(n_outputs, n_inputs)
        self.bias = nn.Parameter(torch.randn(n_outputs))
        self.n_iterations = n_iterations

    @torch.no_grad()
    def power_iteration(self, w):
        # https://arxiv.org/pdf/1804.04368.pdf
        # Spectral Normalization for Generative Adversarial Networks - Miyato et al. - 2018

        # Get maximum singular value of rectangular matrix w
        u = torch.randn(self.n_inputs, 1)
        v = None

        w = w.T

        for _ in range(self.n_iterations):
            wtu = w.T @ u
            v = wtu / torch.linalg.norm(wtu)

            wv = w @ v
            u = wv / torch.linalg.norm(wv)

        factor = u.T @ w @ v
        return factor

    @property
    def normalized_mat(self):
        # Estimate sigma
        sigma = self.power_iteration(self.w)
        # ratio = self.c / sigma
        # return self.w * (ratio ** (ratio < 1))
        return self.w / sigma

    def forward(self, x):
        return torch.nn.functional.linear(x, self.normalized_mat, self.bias)


class SpectralNeuralNetwork(nn.Sequential):
    def __init__(self, n_dim: int, n_hidden: int = 100, n_hidden_layers: int = 2, **kwargs):
        layers = []
        if n_hidden_layers == 0:
            layers = [SpectralLinear(n_dim, n_dim, **kwargs)]
        else:
            layers.append(SpectralLinear(n_dim, n_hidden, **kwargs))
            for _ in range(n_hidden):
                layers.append(nn.Tanh())
                layers.append(SpectralLinear(n_hidden, n_hidden, **kwargs))
            layers.pop(-1)
            layers.append(SpectralLinear(n_hidden, n_dim, **kwargs))
        super().__init__(*layers)


class InvertibleResNet(ResidualBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape)
        self.g = SpectralNeuralNetwork(n_dim=self.n_dim, **kwargs)

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_hutchinson(self.g, x, **kwargs)[1]


class ResFlow(InvertibleResNet):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], p: float = 0.5):
        super().__init__(event_shape)

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_roulette(self.g, x)[1]


class QuasiAutoregressiveFlow(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], sigma: float = 0.7):
        super().__init__(event_shape)
        self.log_theta = nn.Parameter(torch.randn())
        self.sigma = sigma


class ProximalNeuralNetworkBlock(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.b = nn.Parameter(torch.randn(self.n_hidden))
        self.t_tilde = nn.Parameter(torch.randn(self.n_hidden, self.n_inputs))
        self.alpha = ...  # parameters for self.sigma

    @staticmethod
    def sigma(x):
        # sigma is a proximity operator wrt a function g with 0 as a minimizer IFF sigma is 1 lip-cont,
        # monotone increasing and sigma(0) = 0.
        # Tanh qualifies. In fact, many activations do. They are listed in https://arxiv.org/abs/1808.07526v2.
        return torch.tanh(x)

    @property
    def stiefel_matrix(self, n_iterations: int = 5):
        # has shape (n_hidden, n_inputs)
        y = self.t_tilde
        for _ in range(n_iterations):
            y = 2 * y @ torch.linalg.inv(torch.eye(self.n_inputs) + y.T @ y)
        return y

    def regularization(self):
        # to be applied during optimization
        return torch.linalg.norm(self.t_tilde.T @ self.t_tilde - torch.eye(self.n_inputs))

    def forward(self, x):
        mat = self.stiefel_matrix
        act = self.sigma(torch.nn.functional.linear(x, mat, self.b))
        return torch.einsum('...ij,...jk->...ik', mat.T, act)


class ProximalNeuralNetwork(nn.Sequential):
    def __init__(self, n_inputs: int, n_layers: int, n_hidden: int = 100):
        super().__init__(*[ProximalNeuralNetworkBlock(n_inputs, n_hidden) for _ in range(n_layers)])
        self.n_layers = n_layers


class ProximalResidualFlow(ResidualBijection):
    # Note: in its original formulation, this method computes the log of the absolute value of the jacobian determinant
    # using automatic differentiation, which is undesirable. We instead use the hutchinson trace estimator.
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], gamma: float = 1.5, **kwargs):
        super().__init__(event_shape)
        assert gamma > 0
        self.gamma = gamma
        self.pnn = ProximalNeuralNetwork(n_inputs=self.n_dim, **kwargs)
        self.g = lambda x: self.gamma * self.pnn(x)

    def log_det(self, x):
        if self.pnn.n_layers == 1:
            # TODO implement
            raise NotImplementedError
        else:
            return log_det_hutchinson(self.g, x)
