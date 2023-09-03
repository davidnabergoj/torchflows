from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.utils import sum_except_batch


class MonotonicSpline(Transformer):
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
    def n_parameters(self) -> int:
        raise NotImplementedError

    def forward_inputs_inside_bounds_mask(self, x):
        return (x > self.min_input) & (x < self.max_input)

    def inverse_inputs_inside_bounds_mask(self, z):
        return (z > self.min_output) & (z < self.max_output)

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
