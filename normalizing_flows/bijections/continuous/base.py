import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint


# Based on: https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/layers/cnf.py#L11

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class DifferentialEquationNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()


class ConcatenationDiffEq(DifferentialEquationNeuralNetwork):
    # Concatenate t to every layer input
    def __init__(self, event_size: int, hidden_size: int = None, n_hidden_layers: int = 2):
        super().__init__()
        if hidden_size is None:
            hidden_size = max(4, int(3 * math.log10(event_size)))

        self.event_size = event_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

    def forward(self, t, y):
        pass


class ODEFunction(nn.Module):
    def __init__(self, diffeq: DifferentialEquationNeuralNetwork):
        super().__init__()
        self.diffeq = diffeq
        self.register_buffer('_n_evals', torch.tensor(0.0))  # Counts the number of function evaluations
        self._e = None

    def before_odeint(self, e=None):
        self._e = e  # What is e?
        self._n_evals.fill_(0)

    def forward(self, t, states):
        """

        :param t: shape ()
        :param states: (y0, y1, ..., yn) where yi.shape == (batch_size, event_size).
        :return:
        """
        assert len(states) >= 2
        y = states[0]
        self._n_evals += 1

        t = torch.tensor(t).type_as(y)
        batch_size = y.shape[0]

        if self._e is None:
            self._e = torch.randn_like(y)

        with torch.enable_grad():
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            divergence = self.divergence_fn(dy, y, e=self._e).view(batch_size, 1)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class ContinuousFlow(nn.Module):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 f: ODEFunction,
                 end_time: float = 1.0,
                 solver: str = 'dopri5',
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 **kwargs):
        """

        :param event_shape:
        :param f: function to be integrated.
        :param end_time: integrate f from t=0 to t=time_upper_bound. Default: 1.
        :param solver: which solver to use.
        :param kwargs:
        """
        # self.event_shape = event_shape
        self.f = f
        self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(end_time)))
        self.end_time = end_time
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        super().__init__()

    @staticmethod
    def make_default_logpz(z):
        return torch.zeros(size=(z.shape[0], 1)).to(z)

    def make_integrations_times(self, z):
        return torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

    def forward(self,
                x: torch.Tensor,
                logpx=None,
                integration_times=None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if integration_times is None:
            integration_times = self.make_integrations_times(x)
        return self.inverse(x, logpx, _flip(integration_times, 0), **kwargs)

    def inverse(self,
                z: torch.Tensor,
                logpz=None,
                integration_times=None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param z: tensor with shape (batch_size, event_size), i.e. len(x.shape) == 2.
        :param logpz: accumulated log determinant of the jacobian of df/dz.
        :param integration_times:
        :param kwargs:
        :return:
        """
        if logpz is None:
            _logpz = self.make_default_logpz(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = self.make_integrations_times(z)

        # Refresh odefunc statistics
        self.f.before_odeint()

        state_t = odeint(
            self.f,
            (z, _logpz),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t
