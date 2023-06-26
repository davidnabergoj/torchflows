import math
from typing import Tuple
import torch
import torch.nn as nn
from normalizing_flows.src.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.src.utils import get_batch_shape, softmax_nd, sum_except_batch, log_softmax_nd, log_sigmoid


class Combination(Transformer):
    def __init__(self, event_shape: torch.Size, components: list[Transformer]):
        super().__init__(event_shape)
        self.components = components

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        # But this is not maintainable long-term.
        # We probably want ragged tensors with some parameter shape (akin to event and batch shapes).

        # Reshape h for easier access
        n_output_parameters = h.shape[-1] // len(self.components)
        h = torch.stack([
            h[..., i * n_output_parameters:(i + 1) * n_output_parameters]
            for i in range(len(self.components))
        ])

        assert len(h) == len(self.components)

        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(*batch_shape)
        for i in range(len(self.components)):
            x, log_det_increment = self.components[i].forward(x, h[i])
            log_det += log_det_increment

        return x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)

        # Reshape h for easier access
        n_output_parameters = h.shape[-1] // len(self.components)
        h = torch.stack([
            h[..., i * n_output_parameters:(i + 1) * n_output_parameters]
            for i in range(len(self.components))
        ])

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
        # Reshape h for easier access to data
        base_shape = x.shape
        h = h.view(*base_shape, -1, 3)  # (a_unc, b, w_unc)

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


class InverseSigmoidTransform(SigmoidTransform):
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(z, h)


class DenseSigmoidTransform(Transformer):
    def __init__(self,
                 event_shape: torch.Size,
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 8,
                 epsilon: float = 1e-8):
        """

        :param event_shape: ...
        :param hidden_dim: hidden layer dimensionality. Authors recommend 8 or 16.
        """
        super().__init__(event_shape)
        self.hidden_dim = hidden_dim
        self.eps = epsilon

        self.u_ = nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.nn.functional.softplus(h[..., 0])  # Weights must be positive!
        b = h[..., 1]
        w_unc = h[..., 2]
        u_unc = h[..., 3]

        event_dims = tuple(range(len(x.shape)))[-len(self.event_shape):]
        extended_dims = tuple([*event_dims] + [len(x.shape)])
        w = softmax_nd(w_unc, dim=extended_dims)
        u = softmax_nd(u_unc, dim=extended_dims)

        assert a.shape == b.shape == w.shape == u.shape

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


class DeepSigmoidNetwork(Combination):
    """
    Deep sigmoidal network transformer as proposed in "Huang et al. Neural Autoregressive Flows (2018)".
    """

    def __init__(self, event_shape: torch.Size, n_layers: int = 2, **kwargs):
        super().__init__(event_shape, [SigmoidTransform(event_shape, **kwargs) for _ in range(n_layers)])


class InverseDeepSigmoidNetwork(Combination):
    def __init__(self, event_shape: torch.Size, n_layers: int = 2, **kwargs):
        super().__init__(event_shape, [InverseSigmoidTransform(event_shape, **kwargs) for _ in range(n_layers)])
