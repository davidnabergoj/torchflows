from typing import Union, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from normalizing_flows.bijections.finite.base import ConditionalBijection
from normalizing_flows.utils import get_batch_shape, pad_leading_dims


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class TimeDependentNeuralNetwork(nn.Module):
    """
    Neural network that takes as input a scalar t and a tuple of state tensors (y0, y1, ... yn).
    It outputs a predicted tuple of derivatives (dy0/dt, dy1/dt, ... dyn/dt).
    These derivatives determine the ODE system in FFJORD.
    To use time information, t is concatenated to each layer input in this neural network.
    """

    def __init__(self, dim: int, hidden_dim: int = 30):
        super().__init__()
        self.linear1 = nn.Linear(dim + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + 1, dim)

    def forward(self, t: torch.Tensor, y: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        """

        :param t:
        :param y: tuple of tensors (y0, y1, ..., yn). Each yi has shape (*batch_shape, di) where di is the number of
            dimensions for yi.
        :return:
        """
        assert t.shape == ()
        if isinstance(y, torch.Tensor):
            y = (y,)
        n_states = len(y)
        batch_shape = tuple(y[0][:-1])

        # Make t have the same batch shape as yi
        t = pad_leading_dims(t, len(batch_shape))
        t = t.repeat(*batch_shape)
        t = t[..., None]  # Add event dim

        # Recast t into a tuple
        t = tuple([t.clone() for _ in range(n_states)])



        y = torch.cat([y, t], dim=-1)
        y = self.linear1(y)
        y = torch.tanh(y)
        y = self.linear2(y)
        return y


class FFJORD(ConditionalBijection):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 integration_time: float = 1.0,
                 **kwargs):
        super().__init__(event_shape)
        self.g = TimeDependentNeuralNetwork(self.n_dim, **kwargs)
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
