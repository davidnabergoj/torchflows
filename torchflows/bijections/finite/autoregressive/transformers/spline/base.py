from typing import Union, Tuple

import torch

from torchflows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from torchflows.utils import sum_except_batch


class MonotonicSpline(ScalarTransformer):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 min_input: float = -1.0,
                 max_input: float = 1.0,
                 min_output: float = -1.0,
                 max_output: float = 1.0,
                 n_bins: int = 8):
        super().__init__(event_shape)
        self.min_input = min_input
        self.max_input = max_input
        self.min_output = min_output
        self.max_output = max_output
        self.n_bins = n_bins
        self.n_knots = n_bins + 1

    @property
    def parameter_shape_per_element(self) -> int:
        raise NotImplementedError

    def forward_inputs_inside_bounds_mask(self, x):
        return (x > self.min_input) & (x < self.max_input)

    def inverse_inputs_inside_bounds_mask(self, z):
        return (z > self.min_output) & (z < self.max_output)

    def compute_knots_single(self, u, min_size: float, minimum: float, maximum: float) -> torch.Tensor:
        sm = torch.softmax(u, dim=-1)
        spread = min_size + (1 - min_size * self.n_bins) * sm
        knots = torch.cumsum(spread, dim=-1)
        knots = torch.nn.functional.pad(knots, pad=(1, 0), mode='constant', value=0.0)
        knots = (maximum - minimum) * knots + minimum
        knots[..., 0] = minimum
        knots[..., -1] = maximum
        return knots

    def compute_knots(self, u_x, u_y):
        knots_x = self.compute_knots_single(u_x, self.min_width, self.min_input, self.max_input)
        knots_y = self.compute_knots_single(u_y, self.min_height, self.min_output, self.max_output)
        return knots_x, knots_y

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
