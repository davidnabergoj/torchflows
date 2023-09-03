from typing import Union, Tuple

import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class Linear(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], boundary: float = 50.0, **kwargs):
        super().__init__(event_shape,
                         min_input=-boundary,
                         max_input=boundary,
                         min_output=-boundary,
                         max_output=boundary)
        self.u_x = nn.Parameter(torch.randn(size=(self.n_bins,)))
        self.min_width = 1e-3
        self.min_height = 1e-3

    @property
    def bin_x(self):
        sm = torch.softmax(self.u_x, dim=0)
        sm_fixed = self.min_width + (1 - self.min_width * self.n_bins) * sm  # Reduce all sizes and add equal constant
        cs = torch.cumsum(sm_fixed, dim=0)
        cs = torch.nn.functional.pad(cs, (1, 0), value=0.0)
        return cs * (self.max_input - self.min_input) + self.min_input

    def compute_bin_y(self, delta):
        # By setting delta = 0, we recover bin_y = bin_x, which gives the identity transform
        u_y = self.u_x
        sm = torch.softmax(u_y + delta, dim=1)
        sm_fixed = self.min_height + (1 - self.min_height * self.n_bins) * sm  # Reduce all sizes and add equal constant
        cs = torch.cumsum(sm_fixed, dim=1)
        cs = torch.nn.functional.pad(cs, (1, 0), value=0.0)
        return cs * (self.max_output - self.min_output) + self.min_output

    @property
    def n_parameters(self) -> int:
        return self.n_bins

    def forward_1d(self, x, h):
        assert len(x.shape) == 1
        assert len(h.shape) == 2
        assert x.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins

        x_positions = self.bin_x
        bin_indices = torch.searchsorted(x_positions, x) - 1  # Find the correct bin for each input
        bin_widths = x_positions[1:] - x_positions[:-1]
        relative_position_in_bin = (x - x_positions[bin_indices]) / bin_widths[bin_indices]

        y_positions = self.compute_bin_y(delta=h / 1000)
        assert len(y_positions.shape) == 2
        assert x_positions.shape[0] == y_positions.shape[1]

        bin_indices = bin_indices.view(-1, 1)
        bin_floors = y_positions.gather(1, bin_indices)[:, 0]
        bin_ceilings = y_positions.gather(1, bin_indices + 1)[:, 0]
        y = torch.lerp(bin_floors, bin_ceilings, relative_position_in_bin)
        log_det = torch.log(bin_ceilings - bin_floors) - torch.log(bin_widths[bin_indices].flatten())
        return y, log_det

    def inverse_1d(self, y, h):
        assert len(y.shape) == 1
        assert len(h.shape) == 2
        assert y.shape[0] == h.shape[0]
        assert h.shape[1] == self.n_bins
        # Find the correct bin
        # Find relative spot of z in that bin
        # Position = bin_x start + bin_width * relative_spot; this is our output
        y_positions = self.compute_bin_y(delta=h / 1000)
        bin_indices = torch.searchsorted(y_positions, y.view(-1, 1)) - 1  # Find the correct bin for each input
        bin_floors = y_positions.gather(1, bin_indices)[:, 0]
        bin_ceilings = y_positions.gather(1, bin_indices + 1)[:, 0]
        relative_y_position_in_bin = (y - bin_floors) / (bin_ceilings - bin_floors)

        x_positions = self.bin_x
        bin_widths = x_positions[1:] - x_positions[:-1]
        assert len(y_positions.shape) == 2
        assert x_positions.shape[0] == y_positions.shape[1]
        x = x_positions[bin_indices.squeeze(1)] + relative_y_position_in_bin * bin_widths[bin_indices.squeeze(1)]
        log_det = -(torch.log(bin_ceilings - bin_floors) - torch.log(bin_widths[bin_indices.squeeze(1)]))
        return x, log_det
