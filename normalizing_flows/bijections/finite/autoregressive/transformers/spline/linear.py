import math
from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class Linear(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.x_positions = torch.linspace(start=self.min_input, end=self.max_input, steps=self.n_bins)
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
