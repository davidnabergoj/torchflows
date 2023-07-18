import math
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from normalizing_flows.src.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.src.utils import sum_except_batch


class MonotonicPiecewiseSpline(Transformer):
    # With identity extrapolation
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], bound: float = 1.0):
        super().__init__(event_shape)
        self.bound = bound

        self.min_x = -self.bound
        self.max_x = self.bound
        self.min_y = -self.bound
        self.max_y = self.bound

    @staticmethod
    def compute_bin(u, min_val, max_val):
        sm = torch.softmax(u, dim=-1)
        out = min_val + torch.cumsum(sm * (max_val - min_val), dim=-1)
        out = F.pad(out, (1, 0), value=min_val)
        return out

    def compute_bin_x(self, u):
        return self.compute_bin(u, self.min_x, self.max_x)

    def compute_bin_y(self, u):
        return self.compute_bin(u, self.min_y, self.max_y)

    def forward_inputs_inside_bounds_mask(self, x):
        return (x > self.min_x) & (x < self.max_x)

    def inverse_inputs_inside_bounds_mask(self, z):
        return (z > self.min_y) & (z < self.max_y)

    def forward_1d(self, x, h):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.clone(x)  # Remain the same out of bounds
        log_det = torch.zeros_like(z)  # y = x means gradient = 1 or log gradient = 0 out of bounds
        mask = self.forward_inputs_inside_bounds_mask(x)
        if torch.any(mask):
            z[mask], log_det[mask] = self.forward_1d(x[mask], h[mask])
        log_det = sum_except_batch(log_det, event_shape=self.event_shape)
        return z, log_det

    def inverse_1d(self, z, h):
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.clone(z)  # Remain the same out of bounds
        log_det = torch.zeros_like(x)  # y = x means gradient = 1 or log gradient = 0 out of bounds
        mask = self.inverse_inputs_inside_bounds_mask(z)
        if torch.any(mask):
            x[mask], log_det[mask] = self.inverse_1d(z[mask], h[mask])
        log_det = sum_except_batch(log_det, event_shape=self.event_shape)
        return x, log_det


class MonotonicPiecewiseLinearSpline(MonotonicPiecewiseSpline):
    # Monotonic piecewise linear spline with fixed x positions and y=x extrapolation
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_bins: int = 8,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.n_bins = n_bins
        self.x_positions = torch.linspace(start=self.min_x, end=self.max_x, steps=self.n_bins)
        self.w = self.x_positions[1] - self.x_positions[0]

    def forward_1d(self, x, h):
        assert len(x.shape) == 1
        assert len(h.shape) == 2
        assert x.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins
        bin_indices = torch.searchsorted(self.x_positions, x) - 1  # Find the correct bin for each input
        relative_position_in_bin = (x - self.x_positions[bin_indices]) / self.w
        y_positions = self.compute_bin_y(h)

        bin_indices = bin_indices.view(-1, 1)

        bin_floors = y_positions.gather(1, bin_indices)[:, 0]
        bin_ceilings = y_positions.gather(1, bin_indices + 1)[:, 0]

        y = torch.lerp(bin_floors, bin_ceilings, relative_position_in_bin)
        log_det = torch.log(bin_ceilings - bin_floors) - math.log(self.w)
        return y, log_det

    def inverse_1d(self, y, h):
        assert len(y.shape) == 1
        assert len(h.shape) == 2
        assert y.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins

        # Find the correct bin
        # Find relative spot of z in that bin
        # Position = bin_x start + bin_width * relative_spot; this is our output
        y_positions = self.compute_bin_y(h)
        bin_indices = torch.searchsorted(y_positions, y.view(-1, 1)) - 1  # Find the correct bin for each input

        bin_floors = y_positions.gather(1, bin_indices)[:, 0]
        bin_ceilings = y_positions.gather(1, bin_indices + 1)[:, 0]
        relative_y_position_in_bin = (y - bin_floors) / (bin_ceilings - bin_floors)

        x = self.x_positions[bin_indices.squeeze(1)] + relative_y_position_in_bin * self.w
        log_det = -(torch.log(bin_ceilings - bin_floors) - math.log(self.w))
        return x, log_det


class MonotonicPiecewiseQuadraticSpline(MonotonicPiecewiseSpline):
    # C1 continuous
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_bins: int = 8,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.n_bins = n_bins

    @staticmethod
    def compute_quadratic_parameters(x, y, d, idx):
        # Compute a, b, c
        x0 = x.gather(1, idx)[:, 0]
        y0 = y.gather(1, idx)[:, 0]
        d0 = d.gather(1, idx)[:, 0]
        
        x1 = x.gather(1, idx + 1)[:, 0]
        y1 = y.gather(1, idx + 1)[:, 0]
        d1 = d.gather(1, idx + 1)[:, 0]

        a = 0.5 * torch.divide(d0 - d1, x0 - x1)
        b = (a * (x1 ** 2 - x0 ** 2) + y0 - y1) / (x0 - x1)
        # b = d0 - 2 * a * x0
        # assert torch.allclose(b, d1 - 2 * a * x1)
        # assert torch.allclose(b, (a * (x1 ** 2 - x0 ** 2) + y0 - y1) / (x0 - x1))
        c = y0 - a * x0 ** 2 - b * x0
        # assert torch.allclose(c, y1 - a * x1 ** 2 - b * x1)
        return a, b, c

    def forward_1d(self, x, h):
        assert len(x.shape) == 1
        assert len(h.shape) == 2
        assert x.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins * 3 + 2

        x_positions = self.compute_bin_x(h[..., :self.n_bins + 1])
        y_positions = self.compute_bin_y(h[..., self.n_bins + 1:2 * (self.n_bins + 1)])
        derivatives = F.softplus(h[..., 2 * (self.n_bins + 1):])
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)

        bin_indices = torch.searchsorted(x_positions, x.view(-1, 1)) - 1
        a, b, c = self.compute_quadratic_parameters(x_positions, y_positions, derivatives, bin_indices)
        self.plot(h)
        z = (a * x ** 2) + (b * x) + c
        log_det = torch.log(2 * a * x + b)

        return z, log_det

    def plot(self, h):

        x_positions = self.compute_bin_x(h[..., :self.n_bins + 1])
        y_positions = self.compute_bin_y(h[..., self.n_bins + 1:2 * (self.n_bins + 1)])
        derivatives = F.softplus(h[..., 2 * (self.n_bins + 1):])
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)

        x_eval_positions = [self.min_x - 1]
        for i in range(1, len(x_positions[0]) - 1):
            x_eval_positions.append((x_positions[0,i + 1] + x_positions[0,i]) / 2)
        x_eval_positions.append(self.max_x + 1)
        x_eval_positions = torch.tensor(x_eval_positions).view(1, -1)

        for i in range(len(x_eval_positions[0]) - 1):
            xs = torch.linspace(x_eval_positions[0,i], x_eval_positions[0,i + 1], 1000)
            bin_indices = torch.searchsorted(x_positions, x_eval_positions.view(-1, 1)) - 1
            a, b, c = self.compute_quadratic_parameters(x_positions, y_positions, derivatives, bin_indices.ravel())
            plt.plot(xs, a * xs ** 2 + b * xs + c)
        plt.plot([self.max_x, 2 * self.max_x], [self.max_y, 2 * self.max_y])
        plt.plot([self.min_x, 2 * self.min_x], [self.min_y, 2 * self.min_y])
        plt.show()

    def inverse_1d(self, y, h):
        assert len(y.shape) == 1
        assert len(h.shape) == 2
        assert y.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins * 3 + 2

        x_positions = self.compute_bin_x(h[..., :self.n_bins + 1])
        y_positions = self.compute_bin_y(h[..., self.n_bins + 1:2 * (self.n_bins + 1)])
        derivatives = F.softplus(h[..., 2 * (self.n_bins + 1):])
        derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0)

        bin_indices = torch.searchsorted(y_positions, y.view(-1, 1)) - 1

        a, b, c = self.compute_quadratic_parameters(x_positions, y_positions, derivatives, bin_indices)

        sqrt_d = torch.sqrt(b ** 2 - 4 * a * c)
        x = (-b + sqrt_d) / (2 * a)
        x1 = (-b - sqrt_d) / (2 * a)
        mask = (x > x_positions.gather(1, bin_indices)[:, 0])
        x[mask] = x1[mask]

        log_det = -torch.log(2 * a * x + b)
        return x, log_det


# class LinearSpline(Transformer):
#     # Neural importance sampling (2019)
#     # Expects n_bins parameters
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_bins: int = 8):
#         super().__init__(event_shape)
#         self.k = n_bins
#         self.w = 1 / self.k
#
#     def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         q = torch.softmax(h, dim=-1)
#         b = torch.floor(self.k * x).long()
#         alpha = self.k * x - b
#         z = alpha * q[..., b] + torch.cumsum(q, dim=-1)[..., b - 1]
#         log_det = torch.log(q[..., b]) - torch.log(torch.as_tensor(self.w))
#         return z, log_det
#
#     def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         raise NotImplementedError


# class QuadraticSpline(Transformer):
#     # Neural importance sampling (2019)
#     # Expects 2 * n_bins + 1 parameters
#     def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_bins: int = 8):
#         super().__init__(event_shape)
#         self.k = n_bins
#
#     def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         v = h[..., :self.k + 1]
#         w = h[..., self.k + 1:]
#         w = torch.softmax(w, dim=-1)
#
#         exp_v = torch.exp(v)
#         v = exp_v / torch.sum(0.5 * (exp_v[..., :-1] + exp_v[..., 1:]) * w, dim=-1)
#
#         b = ...
#         alpha = x - torch.cumsum(w, dim=-1)[..., b - 1] / w[..., b]
#
#         z = (
#                 alpha ** 2 / 2 * (v[..., b + 1] - v[..., b]) * w[..., b] + alpha * v[..., b] * w[..., b]
#                 + torch.cumsum(0.5 * (v[..., :-1] + v[..., 1:]) * w, dim=-1)[..., b - 1]
#         )
#         log_det = ...
#         return z, log_det
#
#     def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         raise NotImplementedError


class CubicSpline(Transformer):
    # Cubic spline flows
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


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
        self.min_bin_size = 1e-3
        self.min_delta = 1e-5
        self.boundary_u_delta = math.log(math.expm1(1 - self.min_delta))

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
        deltas = self.min_delta + F.softplus(u_deltas)  # Derivatives

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


class LinearRationalSpline(Transformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BSpline(Transformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
