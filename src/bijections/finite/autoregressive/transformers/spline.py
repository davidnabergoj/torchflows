import math
from typing import Tuple
import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer
import torch.nn.functional as F


class RationalQuadraticSpline(Transformer):
    def __init__(self, n_bins: int, boundary: float = 10.0):
        # Neural Spline Flows - Durkan et al. 2019
        super().__init__()
        self.n_bins = n_bins
        self.boundary = boundary
        self.boundary_unconstrained_derivative = math.log(math.expm1(1))

    @staticmethod
    def forward_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, eps=1e-10):
        return torch.subtract(
            2 * torch.log(s_k + eps) +
            torch.log((deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2) + eps),
            2 * torch.log(s_k + (deltas_kp1 + deltas_k - 2 * s_k) * xi_1m_xi + eps)
        )

    @staticmethod
    def rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1):
        log_numerator = 2 * torch.log(s_k) + torch.log(
            deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2
        )
        log_denominator = 2 * torch.log(s_k + term1 * xi_1m_xi)
        log_determinant = log_numerator - log_denominator
        return log_determinant

    def rqs(self, inputs: torch.Tensor, h: torch.Tensor, inverse: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unconstrained spline parameters
        u_widths = h[..., :self.n_bins]
        u_heights = h[..., self.n_bins:2 * self.n_bins]
        u_deltas = h[..., 2 * self.n_bins:]

        u_deltas = torch.nn.functional.pad(u_deltas, pad=(1, 1))
        u_deltas[..., 0] = self.boundary_unconstrained_derivative
        u_deltas[..., -1] = self.boundary_unconstrained_derivative

        assert len(u_widths.shape) == len(u_heights.shape) == len(u_deltas.shape) == 2
        assert u_widths.shape[-1] == u_heights.shape[-1] == self.n_bins == (u_deltas.shape[-1] - 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        # Constrained spline parameters
        widths = torch.softmax(u_widths, dim=-1) * 2 * self.boundary
        heights = torch.softmax(u_heights, dim=-1) * 2 * self.boundary
        deltas = F.softplus(u_deltas)  # Derivatives

        cumulative_widths = F.pad(torch.cumsum(widths, dim=-1), pad=(1, 0), mode='constant', value=0.0)
        cumulative_heights = F.pad(torch.cumsum(heights, dim=-1), pad=(1, 0), mode='constant', value=0.0)

        # Find the correct bin for each input value
        if inverse:
            k = torch.searchsorted(cumulative_heights, inputs[..., None])
        else:
            k = torch.searchsorted(cumulative_widths, inputs[..., None])

        # Index the tensors
        cumulative_heights_k = torch.gather(cumulative_heights, -1, k)
        cumulative_widths_k = torch.gather(cumulative_widths, -1, k)
        heights_k = torch.gather(heights, -1, k)
        widths_k = torch.gather(widths, -1, k)
        deltas_k = torch.gather(deltas, -1, k)
        deltas_kp1 = torch.gather(deltas, -1, k + 1)
        s_k = heights_k / widths_k

        inputs = inputs.view(-1, 1)  # Reshape to facilitate operations
        term1 = deltas_kp1 + deltas_k - 2 * s_k
        if inverse:
            term0 = (inputs - cumulative_heights_k)
            term2 = heights_k * deltas_k

            a = heights_k * s_k - term2 + term0 * term1
            b = term2 - term0 * term1
            c = -s_k * term0

            xi = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
            xi_1m_xi = xi * (1 - xi)

            # Compute the outputs of the inverse pass
            outputs = xi * widths_k + cumulative_widths_k

            # Compute the log determinant of the inverse pass
            log_determinant = self.rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
            return outputs.view(-1), log_determinant.view(-1)
        else:
            xi = (inputs - cumulative_widths_k) / widths_k
            xi_1m_xi = xi * (1 - xi)

            # Compute the outputs of the inverse pass
            numerator0 = heights_k * (s_k * xi ** 2 + deltas_k * xi_1m_xi)
            denominator0 = s_k + term1 * xi_1m_xi
            outputs = cumulative_heights_k + numerator0 / denominator0

            # Compute the log determinant of the inverse pass
            log_determinant = self.rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
            return outputs.view(-1), log_determinant.view(-1)

    def rqs_caller(self, inputs: torch.Tensor, h: torch.Tensor, inverse: bool):
        # Default transform = identity
        outputs = torch.clone(inputs)
        log_determinant = torch.zeros_like(outputs)

        # Set in-bound values according to the RQ spline transformation
        mask = (inputs > -self.boundary) & (inputs < self.boundary)
        outputs[mask], log_determinant[mask] = self.rqs(inputs=inputs[mask], h=h[mask], inverse=inverse)

        return outputs, log_determinant.sum(dim=-1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rqs_caller(x, h, False)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rqs_caller(z, h, True)
