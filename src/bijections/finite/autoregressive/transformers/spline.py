from typing import Tuple
import torch

from src.bijections.finite.autoregressive.transformers.base import Transformer
from src.util import softplus


class RationalQuadraticSpline(Transformer):
    def __init__(self, n_bins: int, boundary: float = 10.0):
        # Neural Spline Flows - Durkan et al. 2019
        super().__init__()
        self.n_bins = n_bins
        self.boundary = boundary

    @staticmethod
    def forward_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi):
        return torch.sum(
            torch.subtract(
                2 * torch.log(s_k) + torch.log((deltas_kp1 * xi ** 2 + 2 * s_k * xi_1m_xi + deltas_k * (1 - xi) ** 2)),
                2 * torch.log(s_k + (deltas_kp1 + deltas_k - 2 * s_k) * xi_1m_xi)
            ),
            dim=-1
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unconstrained spline parameters
        u_widths = h[..., :self.n_bins]
        u_heights = h[..., self.n_bins:2 * self.n_bins]
        u_deltas = h[..., 2 * self.n_bins:]

        assert len(u_widths.shape) == len(u_heights.shape) == len(u_deltas.shape) == 3
        assert u_widths.shape[-1] == u_heights.shape[-1] == self.n_bins == (u_deltas.shape[-1] + 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        # Constrained spline parameters
        widths = torch.softmax(u_widths, dim=-1) * 2 * self.boundary
        heights = torch.softmax(u_heights, dim=-1) * 2 * self.boundary
        deltas = softplus(u_deltas)  # Derivatives

        cumulative_widths = torch.cumsum(widths, dim=-1)
        cumulative_heights = torch.cumsum(heights, dim=-1)

        knots_x = cumulative_widths - self.boundary
        knots_y = cumulative_heights - self.boundary

        # Find the correct bin for each input value
        k = torch.searchsorted(knots_x, x.unsqueeze(2))

        # Index the tensors
        heights_k = torch.index_select(heights.ravel(), -1, k.ravel()).view_as(x)
        widths_k = torch.index_select(widths.ravel(), -1, k.ravel()).view_as(x)
        knots_x_k = torch.index_select(knots_x.ravel(), -1, k.ravel()).view_as(x)
        knots_y_k = torch.index_select(knots_y.ravel(), -1, k.ravel()).view_as(x)
        deltas_k = torch.index_select(deltas.ravel(), -1, k.ravel()).view_as(x)
        deltas_kp1 = torch.index_select(deltas.ravel(), -1, k.ravel() + 1).view_as(x)

        s_k = heights_k / widths_k
        xi = (x - knots_x_k) / widths_k
        xi_1m_xi = xi * (1 - xi)

        z = knots_y_k + torch.divide(
            heights_k * (s_k * xi ** 2 + deltas_k * xi_1m_xi),
            s_k + (deltas_kp1 + deltas_k - 2 * s_k) * xi_1m_xi
        )

        # map_derivatives = torch.divide(
        #     s ** 2 * (deltas_kp1 * xi ** 2 + 2 * s * xi_1m_xi + deltas_k * (1 - xi) ** 2),
        #     (s + (deltas_kp1 + deltas_k - 2 * s) * xi_1m_xi) ** 2
        # )
        # log_determinant = torch.sum(torch.log(map_derivatives), dim=-1)

        # log_determinant = torch.sum(
        #     torch.log(s ** 2 * (deltas_kp1 * xi ** 2 + 2 * s * xi_1m_xi + deltas_k * (1 - xi) ** 2))
        #     - torch.log((s + (deltas_kp1 + deltas_k - 2 * s) * xi_1m_xi) ** 2)
        # )

        log_determinant = self.forward_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi)

        return z, log_determinant

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unconstrained spline parameters
        u_widths = h[..., :self.n_bins]
        u_heights = h[..., self.n_bins:2 * self.n_bins]
        u_deltas = h[..., 2 * self.n_bins:]

        assert len(u_widths.shape) == len(u_heights.shape) == len(u_deltas.shape) == 3
        assert u_widths.shape[-1] == u_heights.shape[-1] == self.n_bins == (u_deltas.shape[-1] + 1)
        # n_data, n_dim, n_transformer_parameters = widths.shape

        # Constrained spline parameters
        widths = torch.softmax(u_widths, dim=-1) * 2 * self.boundary
        heights = torch.softmax(u_heights, dim=-1) * 2 * self.boundary
        deltas = softplus(u_deltas)  # Derivatives

        cumulative_widths = torch.cumsum(widths, dim=-1)
        cumulative_heights = torch.cumsum(heights, dim=-1)

        knots_x = cumulative_widths - self.boundary
        knots_y = cumulative_heights - self.boundary

        # Find the correct bin for each input value
        k = torch.searchsorted(knots_x, z.unsqueeze(2))

        # Index the tensors
        heights_k = torch.index_select(heights.ravel(), -1, k.ravel()).view_as(z)
        widths_k = torch.index_select(widths.ravel(), -1, k.ravel()).view_as(z)
        knots_x_k = torch.index_select(knots_x.ravel(), -1, k.ravel()).view_as(z)
        knots_y_k = torch.index_select(knots_y.ravel(), -1, k.ravel()).view_as(z)
        knots_y_kp1 = torch.index_select(knots_y.ravel(), -1, k.ravel() + 1).view_as(z)
        deltas_k = torch.index_select(deltas.ravel(), -1, k.ravel()).view_as(z)
        deltas_kp1 = torch.index_select(deltas.ravel(), -1, k.ravel() + 1).view_as(z)

        s_k = heights_k / widths_k

        a = (knots_y_kp1 - knots_y_k) * (s_k - deltas_k) + (z - knots_y_k) * (deltas_kp1 + deltas_k - 2 * s_k)
        b = (knots_y_kp1 - knots_y_k) * deltas_k - (z - knots_y_k) * (deltas_kp1 + deltas_k - 2 * s_k)
        c = -s_k * (z - knots_y_k)

        xi = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
        xi_1m_xi = xi * (1 - xi)

        x = xi * widths_k + knots_x_k

        # Opposite of the forward log determinant
        log_determinant = -self.forward_log_determinant(s_k, deltas_k, deltas_kp1, xi, xi_1m_xi)

        return x, log_determinant
