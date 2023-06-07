import math
from typing import Tuple
import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer
import torch.nn.functional as F


class RationalQuadraticSpline(Transformer):
    def __init__(self, event_shape: torch.Size, n_bins: int = 8, boundary: float = 1.0):
        """
        Neural Spline Flows - Durkan et al. 2019

        RQ splines are more prone to numerical instabilities when stacked in a composition than affine transforms.
        This becomes a problem when used in autoregressive flows, since the inverse/forward passes
        (for MAF/IAF respectively) require n_dim spline computations.

        :param n_bins: number of spline bins.
        :param boundary: boundary value for the spline; values outside [-boundary, boundary] remain identical.
        """
        super().__init__(event_shape=event_shape)
        self.n_bins = n_bins
        self.boundary = boundary
        self.boundary_u_delta = math.log(math.expm1(1))
        self.min_bin_size = 1e-3

    @staticmethod
    def rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1):
        log_numerator = 2 * torch.log(s_k) + torch.log1p(
            (deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2) / s_k
        )
        log_denominator = 2 * torch.log(s_k) + torch.log1p(term1 * xi_1m_xi / s_k)
        log_determinant = log_numerator - log_denominator
        return log_determinant

    def compute_bins(self, u, left, right):
        bin_sizes = torch.softmax(u, dim=-1)
        bin_sizes = self.min_bin_size + (1 - self.min_bin_size * self.n_bins) * bin_sizes
        bins = torch.cumsum(bin_sizes, dim=-1)
        bins = F.pad(bins, pad=(1, 0), mode='constant', value=0.0)[..., :-1]
        bins = (right - left) * bins + left
        bins[..., 0] = left
        bins[..., -1] = right
        bin_sizes = bins[..., 1:] - bins[..., :-1]
        return bins, bin_sizes

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

        bin_x, bin_widths = self.compute_bins(u_widths, left, right)
        bin_y, bin_heights = self.compute_bins(u_heights, bottom, top)
        deltas = F.softplus(u_deltas)  # Derivatives

        bin_x = torch.clip(bin_x, left, right)
        bin_y = torch.clip(bin_y, bottom, top)

        assert torch.all(deltas >= 0)

        # Find the correct bin for each input value
        if inverse:
            k = torch.searchsorted(bin_y, inputs[..., None]) - 1
        else:
            k = torch.searchsorted(bin_x, inputs[..., None]) - 1

        assert torch.all(k >= 0)
        assert torch.all(k < self.n_bins), \
            f"{torch.max(k) = }, " \
            f"{float(torch.max(inputs).detach()) = }, " \
            f"{bin_y.shape = }" \
            f"{bin_y[0] = }"

        # Index the tensors
        bin_y_k = torch.gather(bin_y, -1, k)
        bin_x_k = torch.gather(bin_x, -1, k)
        bin_heights_k = torch.gather(bin_heights, -1, k)
        bin_widths_k = torch.gather(bin_widths, -1, k)
        deltas_k = torch.gather(deltas, -1, k)
        deltas_kp1 = torch.gather(deltas, -1, k + 1)
        s_k = bin_heights_k / bin_widths_k

        inputs = inputs.view(-1, 1)  # Reshape to facilitate operations
        term1 = deltas_kp1 + deltas_k - 2 * s_k
        if inverse:
            term0 = (inputs - bin_y_k)
            term2 = bin_heights_k * deltas_k

            a = (bin_heights_k * s_k - term2) + term0 * term1
            b = term2 - term0 * term1
            c = -s_k * term0

            xi = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
            xi_1m_xi = xi * (1 - xi)

            # Compute the outputs of the inverse pass
            outputs = xi * bin_widths_k + bin_x_k

            # Compute the log determinant of the inverse pass
            log_determinant = -self.rqs_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
            return outputs.view(-1), log_determinant.view(-1)
        else:
            xi = (inputs - bin_x_k) / bin_widths_k
            xi_1m_xi = xi * (1 - xi)

            # Compute the outputs of the inverse pass
            numerator0 = bin_heights_k * (s_k * xi ** 2 + deltas_k * xi_1m_xi)
            denominator0 = s_k + term1 * xi_1m_xi
            outputs = bin_y_k + numerator0 / denominator0

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
