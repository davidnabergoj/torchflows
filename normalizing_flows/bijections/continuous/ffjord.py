from typing import Union, Tuple

import torch
from torchdiffeq import odeint_adjoint as odeint

from normalizing_flows.bijections.finite.base import Bijection
from normalizing_flows.utils import get_batch_shape


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(Bijection):
    def __init__(self, g: callable, event_shape: Union[torch.Size, Tuple[int, ...]], integration_time: float = 1.0):
        super().__init__(event_shape)
        self.g = g
        self.integration_time = torch.tensor(integration_time)
        self.register_buffer('sqrt_integration_time', torch.sqrt(self.integration_time))

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None,
                integration_times: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(*get_batch_shape(x, self.event_shape), 1)
        integration_times = torch.tensor([self.integration_time, 0.0])
        state_t = odeint(self.g, (x, log_det), integration_times, **kwargs)
        z, log_det = state_t[:2]
        return z, log_det

    def inverse(self,
                z: torch.Tensor,
                context: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(*get_batch_shape(z, self.event_shape), 1)
        integration_times = torch.tensor([0.0, self.integration_time])
        state_t = odeint(self.g, (z, log_det), integration_times, **kwargs)
        x, log_det = state_t[:2]
        return x, log_det