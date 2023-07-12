from typing import Union, Tuple

import torch

from normalizing_flows.src.bijections import Bijection
from normalizing_flows.src.utils import get_batch_shape


class ResidualBijection(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def g(self, x):
        raise NotImplementedError

    def log_det(self, x):
        raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        x = x.view(*batch_shape, self.n_dim)
        z = x + self.g(x)
        z = z.view(*batch_shape)
        log_det = torch.full(size=batch_shape, fill_value=torch.nan if skip_log_det else self.log_det(x))
        return z, log_det

    def inverse(self,
                z: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False,
                n_iterations: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z = z.view(*batch_shape, self.n_dim)
        x = z
        for _ in range(n_iterations):
            x = z - self.g(x)
        x = x.view(*batch_shape)
        log_det = -torch.full(size=batch_shape, fill_value=torch.nan if skip_log_det else self.log_det(x))
        return x, log_det
