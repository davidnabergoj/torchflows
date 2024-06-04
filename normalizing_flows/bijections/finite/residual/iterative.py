import math
from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.residual.base import ResidualBijection
from normalizing_flows.bijections.finite.residual.log_abs_det_estimators import log_det_power_series, log_det_roulette


class SpectralLinear(nn.Module):
    # https://arxiv.org/pdf/1811.00995.pdf

    def __init__(self, n_inputs: int, n_outputs: int, c: float = 0.7, n_iterations: int = 5):
        super().__init__()
        self.c = c
        self.n_inputs = n_inputs
        self.w = nn.Parameter(torch.randn(n_outputs, n_inputs))
        self.bias = nn.Parameter(torch.randn(n_outputs))
        self.n_iterations = n_iterations

    @torch.no_grad()
    def power_iteration(self, w):
        # https://arxiv.org/pdf/1804.04368.pdf
        # Spectral Normalization for Generative Adversarial Networks - Miyato et al. - 2018

        # Get maximum singular value of rectangular matrix w
        u = torch.randn(self.n_inputs, 1).to(w)
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
    def __init__(self, n_dim: int, n_hidden: int = None, n_hidden_layers: int = 1, **kwargs):
        if n_hidden is None:
            n_hidden = int(3 * max(math.log(n_dim), 4))

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


class InvertibleResNetBlock(ResidualBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], context_shape=None, **kwargs):
        super().__init__(event_shape)
        self.g = SpectralNeuralNetwork(n_dim=self.n_dim, **kwargs)

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_power_series(self.g, x, n_iterations=2, **kwargs)[1]


class ResFlowBlock(InvertibleResNetBlock):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], context_shape=None, p: float = 0.5, **kwargs):
        # TODO add context
        self.p = p
        super().__init__(event_shape, **kwargs)

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_roulette(self.g, x, p=self.p, **kwargs)[1]
