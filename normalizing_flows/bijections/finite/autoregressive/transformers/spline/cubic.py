from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class Cubic(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_bins: int = 8):
        super().__init__(event_shape, n_bins=n_bins)
        self.min_width = 1e-3
        self.min_height = 1e-3
        self.const = 1000

    @property
    def n_parameters(self) -> int:
        return 2 * self.n_bins + 2

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=(self.n_parameters,))

    def compute_spline_parameters(self, knots_x: torch.Tensor, knots_y: torch.Tensor, idx: torch.Tensor):
        # knots_x.shape == (n, n_knots)
        # knots_y.shape == (n, n_knots)
        # idx.shape == (n, 1)

        x_kp2 = knots_x.gather(1, idx + 2)
        x_kp1 = knots_x.gather(1, idx + 1)
        x_k = knots_x.gather(1, idx)
        x_km1 = knots_x.gather(1, idx - 1)
        w_kp1 = x_kp2 - x_kp1
        w_k = x_kp1 - x_k
        w_km1 = x_k - x_km1

        y_kp2 = knots_y.gather(1, idx + 2)
        y_kp1 = knots_y.gather(1, idx + 1)
        y_k = knots_y.gather(1, idx)
        y_km1 = knots_y.gather(1, idx - 1)
        s_kp1 = (y_kp2 - y_kp1) / w_kp1
        s_k = (y_kp1 - y_k) / w_k
        s_km1 = (y_k - y_km1) / w_km1

        p_k = torch.divide(s_km1 * w_k + s_k * w_km1, w_km1 + w_k)
        diff_km1 = torch.minimum(s_km1, s_k)
        mask = p_k > (2 * diff_km1)
        d_k = p_k.clone()
        d_k[mask] = diff_km1[mask]

        p_kp1 = torch.divide(s_k * w_kp1 + s_kp1 * w_k, w_k + w_kp1)
        diff_k = torch.minimum(s_k, s_kp1)
        mask = p_kp1 > (2 * diff_k)
        d_kp1 = p_kp1.clone()
        d_kp1[mask] = diff_k[mask]

        a0 = y_k
        a1 = d_k
        a2 = (3 * s_k - 2 * d_k - d_kp1) / w_k
        a3 = (d_k + d_kp1 - 2 * s_k) / w_k ** 2

        return x_k, a0, a1, a2, a3

    def forward_1d(self, x, h):
        u_x = h[:, :self.n_bins]
        d_y = h[:, self.n_bins:2 * self.n_bins]
        u_y = u_x + d_y / self.const
        knots_x, knots_y = self.compute_knots(u_x, u_y)
        idx = torch.searchsorted(knots_x, x) - 1
        x_k, a0, a1, a2, a3 = self.compute_spline_parameters(knots_x, knots_y, idx)
        xi = x - x_k
        z = a0 + a1 * xi + a2 * xi ** 2 + a3 * xi ** 3

        log_det = ...

        return z, log_det

    def inverse_1d(self, z, h):
        u_x = h[:, :self.n_bins]
        d_y = h[:, self.n_bins:2 * self.n_bins]
        u_y = u_x + d_y / self.const
        knots_x, knots_y = self.compute_knots(u_x, u_y)
