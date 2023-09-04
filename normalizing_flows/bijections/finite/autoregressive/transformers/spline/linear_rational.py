import math
from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class LinearRational(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], boundary: float = 50.0, **kwargs):
        super().__init__(
            event_shape,
            min_input=-boundary,
            max_input=boundary,
            min_output=-boundary,
            max_output=boundary,
            **kwargs
        )
        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_d = 1e-5
        self.const = math.log(math.exp(1 - self.min_d) - 1)  # to ensure identity initialization

    @property
    def n_parameters(self) -> int:
        return 4 * self.n_bins

    def compute_parameters(self, idx, knots_x, knots_y, knots_d, knots_lambda, u_w0):
        assert knots_x.shape == knots_y.shape == knots_d.shape
        assert len(knots_x.shape) == 2
        assert knots_x.shape[1] == self.n_knots
        assert knots_lambda.shape == (knots_x.shape[0], self.n_bins)
        assert idx.shape == (knots_x.shape[0], 1)
        assert u_w0.shape == (knots_x.shape[0],)

        w0 = torch.nn.functional.softplus(u_w0)
        w = w0[:, None] * torch.sqrt(knots_d[:, 0][:, None] / knots_d)
        w_k = w.gather(1, idx)
        w_kp1 = w.gather(1, idx + 1)
        lambda_k = knots_lambda.gather(1, idx)
        x_k = knots_x.gather(1, idx)
        x_kp1 = knots_x.gather(1, idx + 1)
        y_k = knots_y.gather(1, idx)
        y_kp1 = knots_y.gather(1, idx + 1)
        d_k = knots_d.gather(1, idx)
        d_kp1 = knots_d.gather(1, idx + 1)

        x_m = torch.divide(
            (1 - lambda_k) * w_k * x_k + lambda_k * w_kp1 * x_kp1,
            (1 - lambda_k) * w_k + lambda_k * w_kp1
        )  # TODO check that this is correct
        y_m = torch.divide(
            (1 - lambda_k) * w_k * y_k + lambda_k * w_kp1 * y_kp1,
            (1 - lambda_k) * w_k + lambda_k * w_kp1
        )
        w_m = torch.multiply(
            lambda_k * w_k * d_k + (1 - lambda_k) * w_kp1 * d_kp1,
            torch.divide(
                x_kp1 - x_k,
                y_kp1 - y_k
            )
        )
        return lambda_k, w_k, w_m, w_kp1, x_k, x_m, x_kp1, y_k, y_m, y_kp1

    def compute_bins(self, u, minimum, maximum, min_size):
        bin_sizes = torch.softmax(u, dim=-1)
        bin_sizes = min_size + (1 - min_size * self.n_bins) * bin_sizes
        bins = torch.cumsum(bin_sizes, dim=-1)
        bins = torch.nn.functional.pad(bins, pad=(1, 0), mode='constant', value=0.0)
        bins = (maximum - minimum) * bins + minimum
        bins[..., 0] = minimum
        bins[..., -1] = maximum
        return bins

    def compute_derivatives(self, u_d):
        knots_d = torch.nn.functional.softplus(u_d) + self.min_d
        knots_d = torch.nn.functional.pad(knots_d, pad=(1, 1), mode='constant', value=1.0)
        return knots_d

    def compute_knots(self, u_x, u_y, u_l, u_d):
        # u_y acts as a delta
        # u_d acts as a delta
        knots_x = self.compute_bins(u_x, self.min_input, self.max_input, self.min_bin_width)
        knots_y = self.compute_bins(u_x + u_y / 1000, self.min_output, self.max_output, self.min_bin_height)
        knots_lambda = torch.sigmoid(u_l)
        knots_d = self.compute_derivatives(self.const + u_d / 1000)
        return knots_x, knots_y, knots_d, knots_lambda

    def forward_1d(self, x, h):
        assert len(x.shape) == 1
        assert len(h.shape) == 2
        assert h.shape[0] == x.shape[0]
        u_x = h[:, :self.n_bins]
        u_y = h[:, self.n_bins:2 * self.n_bins]
        u_lambda = h[:, 2 * self.n_bins:3 * self.n_bins]
        u_d = h[:, 3 * self.n_bins:4 * self.n_bins - 1]
        u_w0 = h[:, 4 * self.n_bins - 1]

        knots_x, knots_y, knots_d, knots_lambda = self.compute_knots(u_x, u_y, u_lambda, u_d)
        idx = torch.searchsorted(knots_x, x[:, None]) - 1
        lambda_k, w_k, w_m, w_kp1, x_k, x_m, x_kp1, y_k, y_m, y_kp1 = self.compute_parameters(
            idx, knots_x, knots_y, knots_d, knots_lambda, u_w0
        )

        phi = (x[:, None] - x_k) / (x_kp1 - x_k)
        mask = phi > lambda_k

        outputs_phi_lt_lambda = torch.divide(
            w_k * y_k * (lambda_k - phi) + w_m * y_m * phi,
            w_k * (lambda_k - phi) + w_m * phi
        ) * (x_kp1 - x_k) + x_k
        log_det_phi_lt_lambda = (
                torch.log(lambda_k * w_k * w_m * (y_m - y_k))
                - 2 * torch.log(w_k * (lambda_k - phi) + w_m * phi)
                - torch.log(x_kp1 - x_k)
        )

        outputs_phi_gt_lambda = torch.divide(
            w_m * y_m * (1 - phi) + w_kp1 * y_kp1 * (phi - lambda_k),
            w_m * (1 - phi) + w_kp1 * (phi - lambda_k)
        ) * (x_kp1 - x_k) + x_k
        log_det_phi_gt_lambda = (
                torch.log((1 - lambda_k) * w_m * w_kp1 * (y_kp1 - y_m))
                - 2 * torch.log(w_m * (1 - phi) + w_m * (phi - lambda_k))
                - torch.log(x_kp1 - x_k)
        )

        outputs = outputs_phi_lt_lambda.flatten()
        outputs[mask[:, 0]] = outputs_phi_gt_lambda[mask]

        log_det = log_det_phi_lt_lambda.flatten()
        log_det[mask[:, 0]] = log_det_phi_gt_lambda[mask]

        return outputs, log_det

    def inverse_1d(self, z, h):
        assert len(z.shape) == 1
        assert len(h.shape) == 2
        assert h.shape[0] == z.shape[0]
        u_x = h[:, :self.n_bins]
        u_y = h[:, self.n_bins:2 * self.n_bins]
        u_lambda = h[:, 2 * self.n_bins:3 * self.n_bins]
        u_d = h[:, 3 * self.n_bins:4 * self.n_bins - 1]
        u_w0 = h[:, 4 * self.n_bins - 1]

        knots_x, knots_y, knots_d, knots_lambda = self.compute_knots(u_x, u_y, u_lambda, u_d)
        idx = torch.searchsorted(knots_y, z[:, None]) - 1
        lambda_k, w_k, w_m, w_kp1, x_k, x_m, x_kp1, y_k, y_m, y_kp1 = self.compute_parameters(
            idx, knots_x, knots_y, knots_d, knots_lambda, u_w0
        )

        z = z[:, None]
        mask = z > y_m
        outputs_y_lt_ym = torch.divide(
            lambda_k * w_k * (y_k - z),
            w_k * (y_k - z) + w_m * (z - y_m)
        )
        log_det_y_lt_ym = (
                torch.log(lambda_k * w_k * w_m * (y_m - y_k))
                - 2 * torch.log(w_k * (y_k - z) + w_m * (z - y_m))
                + torch.log(x_kp1 - x_k)
        )

        outputs_y_gt_ym = torch.divide(
            lambda_k * w_kp1 * (y_kp1 - z) + w_m * (z - y_m),
            w_kp1 * (y_kp1 - z) + w_m * (z - y_m)
        )
        log_det_y_gt_ym = (
                torch.log((1 - lambda_k) * w_m * w_kp1 * (y_kp1 - y_m))
                - 2 * torch.log(w_kp1 * (y_kp1 - z) + w_m * (z - y_m))
                + torch.log(x_kp1 - x_k)
        )

        outputs = outputs_y_lt_ym.flatten()
        outputs[mask[:, 0]] = outputs_y_gt_ym[mask]

        log_det = log_det_y_lt_ym.flatten()
        log_det[mask[:, 0]] = log_det_y_gt_ym[mask]

        return outputs, log_det
