from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.types import _size


class ConcatenatedDistribution(torch.distributions.Distribution, nn.Module):
    def __init__(self, distributions: List[torch.distributions.Distribution]):
        for d in distributions:
            if len(d.event_shape) != 1:
                raise ValueError(f"Component must have 1 event dimension, found {d.event_shape}")
        concatenated_event_shape = (sum([d.event_shape[0] for d in distributions]),)
        super().__init__(event_shape=torch.Size(concatenated_event_shape), validate_args=False)
        self.distributions = distributions

        self.distribution_n_dims = [d.event_shape[0] for d in distributions]
        cs = [0] + np.cumsum(self.distribution_n_dims).tolist()
        self.distribution_masks = []
        for i in range(len(self.distribution_n_dims)):
            mask = (cs[i] <= torch.arange(self.event_shape[0])) & (torch.arange(self.event_shape[0]) < cs[i + 1])
            self.distribution_masks.append(mask)

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        return torch.concat([
            d.sample(sample_shape)
            for d in self.distributions
        ], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.concat([
            d.log_prob(value[..., m])[..., None]
            for m, d in zip(self.distribution_masks, self.distributions)
        ], dim=-1).sum(dim=-1)
