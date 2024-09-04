import math
from typing import Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.finite.residual.base import IterativeResidualBijection
from torchflows.bijections.finite.residual.log_abs_det_estimators import log_det_power_series, log_det_roulette
from torchflows.utils import get_batch_shape


class SpectralMatrix(nn.Module):
    def __init__(self, shape: Tuple[int, int], c: float = 0.7, n_iterations: int = 5):
        super().__init__()
        self.data = nn.Parameter(torch.randn(size=shape))
        self.c = c
        self.n_iterations = n_iterations

    @torch.no_grad()
    def power_iteration(self, w):
        # https://arxiv.org/pdf/1804.04368.pdf
        # Spectral Normalization for Generative Adversarial Networks - Miyato et al. - 2018

        # Get maximum singular value of rectangular matrix w
        u = torch.randn(self.data.shape[1], 1).to(w)
        v = None

        w = w.T

        for _ in range(self.n_iterations):
            wtu = w.T @ u
            v = wtu / torch.linalg.norm(wtu)

            wv = w @ v
            u = wv / torch.linalg.norm(wv)

        factor = u.T @ w @ v
        return factor

    def normalized(self):
        # Estimate sigma
        with torch.no_grad():
            sigma = self.power_iteration(self.data)
        return self.data / sigma


class SpectralLinear(nn.Module):
    # https://arxiv.org/pdf/1811.00995.pdf

    def __init__(self, n_inputs: int, n_outputs: int, **kwargs):
        super().__init__()
        self.w = SpectralMatrix((n_outputs, n_inputs), **kwargs)
        self.bias = nn.Parameter(torch.randn(n_outputs))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.w.normalized(), self.bias)


class SpectralConv2d(nn.Module):
    def __init__(self, n_channels: int, kernel_shape: Tuple[int, int] = (3, 3), **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_shape = kernel_shape
        self.weight = SpectralMatrix((n_channels * kernel_shape[0], n_channels * kernel_shape[1]), **kwargs)
        self.bias = nn.Parameter(torch.randn(n_channels))

    def forward(self, x):
        w = self.weight.normalized().view(self.n_channels, self.n_channels, *self.kernel_shape)
        return torch.conv2d(x, w, self.bias, padding='same')


class SpectralNeuralNetwork(nn.Module):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], torch.Size],
                 n_hidden: int = None,
                 n_hidden_layers: int = 1,
                 **kwargs):
        self.event_shape = event_shape
        event_size = int(torch.prod(torch.as_tensor(event_shape)))
        if n_hidden is None:
            n_hidden = int(3 * max(math.log(event_size), 4))

        if n_hidden_layers == 0:
            layers = [SpectralLinear(event_size, event_size, **kwargs)]
        else:
            layers = [SpectralLinear(event_size, n_hidden, **kwargs)]
            for _ in range(n_hidden):
                layers.append(nn.Tanh())
                layers.append(SpectralLinear(n_hidden, n_hidden, **kwargs))
            layers.pop(-1)
            layers.append(SpectralLinear(n_hidden, event_size, **kwargs))
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        batch_shape = get_batch_shape(x, self.event_shape)
        x_flat = x.view(*batch_shape, -1)
        for layer in self.layers:
            x_flat = layer(x_flat)
        x = x_flat.view_as(x)
        return x


class SpectralCNN(nn.Sequential):
    def __init__(self, n_channels: int, n_layers: int = 2, **kwargs):
        layers = []
        for _ in range(n_layers):
            layers.append(SpectralConv2d(n_channels, **kwargs))
            layers.append(nn.Tanh())
        layers.pop(-1)
        super().__init__(*layers)


class InvertibleResNetBlock(IterativeResidualBijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 g: nn.Module = None,
                 **kwargs):
        # TODO add context
        super().__init__(event_shape)
        if g is None:
            g = SpectralNeuralNetwork(event_shape, **kwargs)
        self.g = g

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_power_series(self.event_shape, self.g, x, n_iterations=2, **kwargs)[1]


class ConvolutionalInvertibleResNetBlock(InvertibleResNetBlock):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 **kwargs):
        # TODO add context
        super().__init__(event_shape, g=SpectralCNN(n_channels=event_shape[0]), **kwargs)


class ResFlowBlock(IterativeResidualBijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 g: nn.Module = None,
                 p: float = 0.5,
                 **kwargs):
        # TODO add context
        super().__init__(event_shape)
        if g is None:
            g = SpectralNeuralNetwork(event_shape, **kwargs)
        self.g = g
        self.p = p

    def log_det(self, x: torch.Tensor, **kwargs):
        return log_det_roulette(self.event_shape, self.g, x, p=self.p, **kwargs)[1]


class ConvolutionalResFlowBlock(ResFlowBlock):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 **kwargs):
        # TODO add context
        super().__init__(event_shape, g=SpectralCNN(n_channels=event_shape[0]), **kwargs)
