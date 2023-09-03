from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline
import torch.nn as nn


class LinearRational(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.unconstrained_w0 = nn.Parameter(torch.tensor(0.0))
        self.unconstrained_lambda = nn.Parameter(torch.zeros(size=(self.n_bins,)))

    @property
    def w0(self):
        return torch.nn.functional.softplus(self.unconstrained_w0)

    @property
    def lmbd(self):
        return torch.sigmoid(self.unconstrained_lambda)

    def compute_parameters(self, h):
        # We have knots (x, y, delta), but the derivatives at boundary knots are fixed to 1 (linear extrapolation)
        assert h.shape[-1] == 3 * self.n_knots - 2

        # Process inputs
        # TODO put knot parsing into an inherited method, defined in MonotonicPiecewiseSpline
        x = h[:self.n_knots]  # TODO cumsum etc
        y = h[self.n_knots:2 * self.n_knots]  # TODO cumsum etc
        delta = torch.nn.functional.softplus(h[2 * self.n_knots:])

        batch_shape = h.shape[:-1]

        # Compute w
        w0 = (torch.zeros(size=batch_shape) + self.w0)[..., None]
        delta_sqrt_ratios = torch.cat([w0, torch.sqrt(delta[..., :-1] / delta[..., 1:])], dim=-1)
        w = torch.cumprod(delta_sqrt_ratios, dim=-1)

        # Compute y(m)
        lmbd = ...  # TODO repeat self.lmbd to batch shape
        y_middle = torch.divide(
            (1 - lmbd) * w[..., :-1] * y[..., :-1] + lmbd * w[..., 1:] * y[..., 1:],
            (1 - lmbd) * w[..., :-1] + lmbd * w[..., 1:]
        )

        # Compute w(m)
        s = (x[..., 1:] - x[..., -1:]) / (y[..., 1:] - y[..., -1:])
        w_middle = s * torch.add(
            lmbd * w[..., :-1] * delta[..., :-1],
            (1 - lmbd) * w[..., 1:] * delta[..., 1:]
        )

        return x, y, delta, w, w_middle, y_middle
