import math
from typing import Tuple
import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer
import torch.nn.functional as F


class RationalQuadraticSpline(Transformer):
    # Increasing these values will increase reconstruction error
    min_bin_width = 1e-6
    min_bin_height = 1e-6
    min_delta = 1e-6

    def __init__(self, n_bins: int = 8, boundary: float = 1.0):
        # Neural Spline Flows - Durkan et al. 2019
        super().__init__()
        self.n_bins = n_bins
        self.boundary = boundary
        self.boundary_u_delta = math.log(math.expm1(1 - self.min_delta))

    @staticmethod
    def rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1):
        log_numerator = torch.add(
            2 * torch.log(s_k),
            torch.log(deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2)
        )
        log_denominator = 2 * torch.log(s_k + term1 * xi_1m_xi)
        log_determinant = log_numerator - log_denominator
        return log_determinant

    def compute_cumulative_bins(self, u, interval_left, interval_right, min_bin_size):
        bins = min_bin_size + (1 - min_bin_size * self.n_bins) * torch.softmax(u, dim=-1)
        cumulative_bins = F.pad(torch.cumsum(bins, dim=-1), pad=(1, 0), mode='constant', value=0.0)
        cumulative_bins = interval_left + (interval_right - interval_left) * cumulative_bins
        cumulative_bins[..., 0] = interval_left
        cumulative_bins[..., -1] = interval_right
        bins = cumulative_bins[..., 1:] - cumulative_bins[..., :-1]
        return bins, cumulative_bins

    def rqs(self,
            inputs: torch.Tensor,
            u_widths: torch.Tensor,
            u_heights: torch.Tensor,
            u_deltas: torch.Tensor,
            inverse: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        left = bottom = -self.boundary
        right = top = self.boundary

        assert (
                (inverse and torch.all((inputs >= bottom) & (inputs <= top))) or
                (not inverse and torch.all((inputs >= left) & (inputs <= right)))
        )
        assert len(u_widths.shape) == len(u_heights.shape) == len(u_deltas.shape) == 2
        assert u_widths.shape[-1] == u_heights.shape[-1] == self.n_bins == (u_deltas.shape[-1] - 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        widths, cumulative_widths = self.compute_cumulative_bins(u_widths, left, right, self.min_bin_width)
        heights, cumulative_heights = self.compute_cumulative_bins(u_heights, bottom, top, self.min_bin_height)
        deltas = self.min_delta + F.softplus(u_deltas)  # Derivatives

        # Find the correct bin for each input value
        if inverse:
            k = torch.searchsorted(cumulative_heights, inputs[..., None]) - 1
        else:
            k = torch.searchsorted(cumulative_widths, inputs[..., None]) - 1

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
            log_determinant = -self.rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
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

        # Unconstrained spline parameters
        u_widths = h[..., :self.n_bins]
        u_heights = h[..., self.n_bins:2 * self.n_bins]
        u_deltas = F.pad(h[..., 2 * self.n_bins:], pad=(1, 1), mode='constant', value=self.boundary_u_delta)

        # Set in-bound values according to the RQ spline transformation
        mask = (inputs > -self.boundary) & (inputs < self.boundary)
        outputs[mask], log_determinant[mask] = self.rqs(
            inputs=inputs[mask],
            u_widths=u_widths[mask],
            u_heights=u_heights[mask],
            u_deltas=u_deltas[mask],
            inverse=inverse
        )

        return outputs, log_determinant.sum(dim=-1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rqs_caller(x, h, False)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rqs_caller(z, h, True)
