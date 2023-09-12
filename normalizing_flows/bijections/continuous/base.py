from typing import Union, Tuple, List

import torch
import torch.nn as nn
from torchdiffeq import odeint
from normalizing_flows.bijections.continuous.layers import DiffEqLayer
import normalizing_flows.bijections.continuous.layers as diff_eq_layers


# Based on: https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/layers/cnf.py#L11

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def create_nn(event_size: int, hidden_size: int = 30, n_hidden_layers: int = 2):
    assert n_hidden_layers >= 0
    if n_hidden_layers == 0:
        layers = [diff_eq_layers.ConcatLinear(event_size, event_size)]
    else:
        layers = [
            diff_eq_layers.ConcatLinear(event_size, hidden_size),
            *[diff_eq_layers.ConcatLinear(hidden_size, hidden_size) for _ in range(n_hidden_layers)],
            diff_eq_layers.ConcatLinear(hidden_size, event_size)
        ]

    return DifferentialEquationNeuralNetwork(layers)


class DifferentialEquationNeuralNetwork(nn.Module):
    """
    Neural network that takes as input a scalar t and a tuple of state tensors (y0, y1, ... yn).
    It outputs a predicted tuple of derivatives (dy0/dt, dy1/dt, ... dyn/dt).
    These derivatives determine the ODE system in FFJORD.
    To use time information, t is concatenated to each layer input in this neural network.
    """

    def __init__(self, layers: List[DiffEqLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, t, x):
        # Reshape t and x
        dx = x
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)

            # Apply nonlinearity
            if i < len(self.layers) - 1:
                dx = torch.tanh(dx)
        # Reshape back
        return dx


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
            divergence = divergence_approx(dy, y, e=self._e).view(batch_size, 1)
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
        super().__init__()
        self.event_shape = event_shape
        self.n_dim = int(torch.prod(torch.as_tensor(self.event_shape)))
        self.f = f
        self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(end_time)))
        self.end_time = end_time
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def make_integrations_times(self, z):
        return torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

    def inverse(self,
                z: torch.Tensor,
                integration_times: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param z: tensor with shape (batch_size, event_size), i.e. len(x.shape) == 2.
        :param integration_times:
        :param kwargs:
        :return:
        """

        if integration_times is None:
            integration_times = self.make_integrations_times(z)

        # Refresh odefunc statistics
        self.f.before_odeint()

        log_det_initial = torch.zeros(size=(z.shape[0], 1)).to(z)
        state_t = odeint(
            self.f,
            (z, log_det_initial),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_final, log_det_final = state_t[:2]

        x = z_final
        return x, log_det_final

    def forward(self,
                x: torch.Tensor,
                integration_times: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if integration_times is None:
            integration_times = self.make_integrations_times(x)
        return self.inverse(
            x,
            integration_times=_flip(integration_times, 0),
            **kwargs
        )
