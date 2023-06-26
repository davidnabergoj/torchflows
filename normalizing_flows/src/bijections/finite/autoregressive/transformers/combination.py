import math
from typing import Tuple
import torch
from normalizing_flows.src.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.src.utils import get_batch_shape, softmax_nd, sum_except_batch, log_softmax_nd, log_sigmoid, \
    logsumexp_nd


class Combination(Transformer):
    def __init__(self, event_shape: torch.Size, components: list[Transformer]):
        super().__init__(event_shape)
        self.components = components

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (n_components, *batch_size, *event_shape)
        assert len(h) == len(self.components)

        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(*batch_shape)
        for i in range(len(self.components)):
            x, log_det_increment = self.components[i].forward(x, h[i])
            log_det += log_det_increment

        return x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (n_components, *batch_size, *event_shape)
        assert len(h) == len(self.components)

        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(*batch_shape)
        for i in range(len(self.components) - 1, -1, -1):
            z, log_det_increment = self.components[i].inverse(z, h[i])
            log_det += log_det_increment

        return z, log_det


class SigmoidTransform(Transformer):
    """
    Smallest invertible component of the deep sigmoidal networks.
    """

    def __init__(self, event_shape: torch.Size, hidden_dim: int = 8, epsilon: float = 1e-8):
        """

        :param event_shape: ...
        :param hidden_dim: hidden layer dimensionality. Authors recommend 8 or 16.
        """
        super().__init__(event_shape)
        self.hidden_dim = hidden_dim
        self.eps = epsilon

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward transformation is equal to y = log(w.T @ sigmoid(a * x + b)).

        :param x: inputs
        :param h: transformation parameters
        :return: outputs and log of the Jacobian determinant
        """
        a = torch.nn.functional.softplus(h[..., 0])  # Weights must be positive!
        b = h[..., 1]
        w_unc = h[..., 2]

        event_dims = tuple(range(len(x.shape)))[-len(self.event_shape):]
        extended_dims = tuple([*event_dims] + [len(x.shape)])
        w = softmax_nd(w_unc, dim=extended_dims)

        assert a.shape == b.shape == w.shape

        x_unsqueezed = x[..., None]  # Unsqueeze last dimension
        x_affine = a * x_unsqueezed + b
        x_sigmoid = torch.sigmoid(x_affine)
        x_convex = torch.sum(w * x_sigmoid, dim=-1)  # Sum over aux dim (dot product)
        x_convex_clipped = x_convex * (1 - self.eps) + self.eps * 0.5
        y = torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)

        log_det = log_softmax_nd(w_unc, extended_dims) + log_sigmoid(x_affine) + log_sigmoid(-x_affine) + torch.log(a)
        log_det = torch.logsumexp(log_det, -1)  # LSE over aux dim
        log_det += math.log(1 - self.eps) - torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)
        log_det = sum_except_batch(log_det, self.event_shape)
        # log_det = sum_except_batch(torch.log(torch.sum((1 - x_sigmoid) * a, dim=-1)), self.event_shape)
        return y, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The original paper introducing deep sigmoidal networks did not provide an analytic inverse.
        # Inverting the transformation can be done numerically.
        raise NotImplementedError


class DeepSigmoidalNetwork(Combination):
    """
    Deep sigmoidal network transformer as proposed in "Huang et al. Neural Autoregressive Flows (2018)".
    """

    def __init__(self, event_shape: torch.Size, n_hidden_layers: int = 2, **kwargs):
        super().__init__(event_shape, [SigmoidTransform(event_shape, **kwargs) for _ in range(n_hidden_layers)])
