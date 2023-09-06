import math
from typing import Union, Tuple

import torch
import torch.nn.functional as F

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class RationalQuadratic(MonotonicSpline):
    """
    Neural Spline Flows - Durkan et al. 2019
    RQ splines are more prone to numerical instabilities when stacked in a composition than affine transforms.
    This becomes a problem when used in autoregressive flows, since the inverse/forward passes
    (for MAF/IAF respectively) require n_dim spline computations.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 boundary: float = 50.0,
                 **kwargs):
        """

        :param event_shape:
        :param boundary: boundary value for the spline; values outside [-boundary, boundary] remain identical.
        :param kwargs:
        """
        super().__init__(
            event_shape,
            min_input=-boundary,
            max_input=boundary,
            min_output=-boundary,
            max_output=boundary,
            **kwargs
        )
        self.min_bin_size = 1e-3
        self.min_delta = 1e-5
        self.boundary_u_delta = math.log(math.expm1(1 - self.min_delta))

    @property
    def n_parameters(self) -> int:
        return 3 * self.n_bins - 1

    @property
    def default_parameters(self) -> torch.Tensor:
        default_u_x = torch.zeros(size=(self.n_bins,))
        default_u_y = torch.zeros(size=(self.n_bins,))
        default_u_d = torch.zeros(size=(self.n_bins - 1,))
        return torch.cat([default_u_x, default_u_y, default_u_d], dim=0)

    def compute_bins(self, u, minimum, maximum):
        bin_sizes = torch.softmax(u, dim=-1)
        bin_sizes = self.min_bin_size + (1 - self.min_bin_size * self.n_bins) * bin_sizes
        bins = torch.cumsum(bin_sizes, dim=-1)
        bins = F.pad(bins, pad=(1, 0), mode='constant', value=0.0)
        bins = (maximum - minimum) * bins + minimum
        bins[..., 0] = minimum
        bins[..., -1] = maximum
        bin_sizes = bins[..., 1:] - bins[..., :-1]
        return bins, bin_sizes

    @staticmethod
    def log_det(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1):
        log_numerator = 2 * torch.log(s_k) + torch.log(
            (deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2)
        )
        log_denominator = 2 * torch.log(s_k + term1 * xi_1m_xi)
        log_determinant = log_numerator - log_denominator
        return log_determinant

    def rqs_forward_1d(self,
                       inputs: torch.Tensor,
                       u_x: torch.Tensor,
                       u_y: torch.Tensor,
                       u_d: torch.Tensor):
        assert torch.all(torch.as_tensor(inputs >= self.min_input) & torch.as_tensor(inputs <= self.max_input))
        assert len(u_x.shape) == len(u_y.shape) == len(u_d.shape) == 2
        assert u_x.shape[-1] == u_y.shape[-1] == self.n_bins == (u_d.shape[-1] - 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        bin_x, bin_widths = self.compute_bins(u_x, self.min_input, self.max_input)
        bin_y, bin_heights = self.compute_bins(u_x + u_y / 1000, self.min_output, self.max_output)
        deltas = self.min_delta + F.softplus(self.boundary_u_delta + u_d / 1000)  # Derivatives

        assert torch.all(deltas >= 0)

        # Find the correct bin for each input value
        k = torch.searchsorted(bin_x, inputs[..., None]) - 1

        assert torch.all(k >= 0)
        assert torch.all(k < self.n_bins)

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

        xi = (inputs - bin_x_k) / bin_widths_k
        xi_1m_xi = xi * (1 - xi)

        # Compute the outputs of the inverse pass
        numerator0 = bin_heights_k * (s_k * xi ** 2 + deltas_k * xi_1m_xi)
        denominator0 = s_k + term1 * xi_1m_xi
        outputs = bin_y_k + numerator0 / denominator0

        # Compute the log determinant of the inverse pass
        log_determinant = self.log_det(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
        return outputs.view(-1), log_determinant.view(-1)

    def forward_1d(self, x, h):
        """

        :param x: torch.Tensor with shape (n_data,)
        :param h: torch.Tensor with shape (n_data, n_spline_parameters)
        :return:
        """
        assert len(x.shape) == 1
        assert len(h.shape) == 2
        assert x.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins * 3 - 1

        # Unconstrained spline parameters
        u_x = h[..., :self.n_bins]
        u_y = h[..., self.n_bins:2 * self.n_bins]
        u_d = F.pad(h[..., 2 * self.n_bins:], pad=(1, 1), mode='constant', value=self.boundary_u_delta)
        return self.rqs_forward_1d(x, u_x, u_y, u_d)

    def rqs_inverse_1d(self,
                       inputs: torch.Tensor,
                       u_x: torch.Tensor,
                       u_y: torch.Tensor,
                       u_d: torch.Tensor):
        assert torch.all(torch.as_tensor(inputs >= self.min_output) & torch.as_tensor(inputs <= self.max_output))
        assert len(u_x.shape) == len(u_y.shape) == len(u_d.shape) == 2
        assert u_x.shape[-1] == u_y.shape[-1] == self.n_bins == (u_d.shape[-1] - 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        bin_x, bin_widths = self.compute_bins(u_x, self.min_input, self.max_input)
        bin_y, bin_heights = self.compute_bins(u_x + u_y / 1000, self.min_output, self.max_output)
        deltas = self.min_delta + F.softplus(self.boundary_u_delta + u_d / 1000)  # Derivatives

        assert torch.all(deltas >= 0)

        # Find the correct bin for each input value
        k = torch.searchsorted(bin_y, inputs[..., None]) - 1

        assert torch.all(k >= 0)
        assert torch.all(k < self.n_bins)

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
        log_determinant = -self.log_det(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi, term1)
        return outputs.view(-1), log_determinant.view(-1)

    def inverse_1d(self, z, h):
        """

        :param z: torch.Tensor with shape (n_data,)
        :param h: torch.Tensor with shape (n_data, n_spline_parameters)
        :return:
        """
        assert len(z.shape) == 1
        assert len(h.shape) == 2
        assert z.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins * 3 - 1

        # Unconstrained spline parameters
        u_x = h[..., :self.n_bins]
        u_y = h[..., self.n_bins:2 * self.n_bins]
        u_d = F.pad(h[..., 2 * self.n_bins:], pad=(1, 1), mode='constant', value=self.boundary_u_delta)
        return self.rqs_inverse_1d(z, u_x, u_y, u_d)
