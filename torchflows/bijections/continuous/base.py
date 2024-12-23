# This file incorporates work covered by the following copyright and permission notice:
#
#   MIT License
#
#   Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.


# This file is an adaptation of code from the following repository https://github.com/rtqichen/ffjord

import math
from typing import Union, Tuple, List

import torch
import torch.nn as nn

from torchflows.bijections.base import Bijection
from torchflows.bijections.continuous.layers import DiffEqLayer, ConcatConv2d, IgnoreConv2d
import torchflows.bijections.continuous.layers as diff_eq_layers
from torchflows.utils import flatten_event, flatten_batch, get_batch_shape, unflatten_batch, unflatten_event


# TODO: have ODEFunction and RegularizedODEFunction return reg_states as the third output.
#       This should be an expected output in tests.
#       We should create a ContinuousFlow class which handles the third output and uses it in the fit method
#       Alternatively: store reg_states as an attribute of the bijection. Make it so that the base bijection class
#       contains a reg_states attribute, which is accessed during Flow.fit.


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def divergence_approx_basic(f, y, e: torch.Tensor = None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_approx_extended(f, y, e: Union[torch.Tensor, Tuple[torch.Tensor]] = None):
    """

    :param e: tuple of noise samples for hutchinson trace estimation.
              One noise tensor may be enough for unbiased estimates.
    :return: divergence and frobenius norms of jacobians.
    """
    if isinstance(e, torch.Tensor):
        e = (e,)  # Convert to tuple

    samples = []
    sqnorms = []
    for e_ in e:
        e_dzdx = torch.autograd.grad(f, y, e_, create_graph=True)[0]
        n = e_dzdx.view(y.size(0), -1).pow(2).mean(dim=1, keepdim=True)
        sqnorms.append(n)
        e_dzdx_e = e_dzdx * e_
        samples.append(e_dzdx_e.view(y.shape[0], -1).sum(dim=1, keepdim=True))
    S = torch.cat(samples, dim=1)
    approx_tr_dzdx = S.mean(dim=1)
    N = torch.cat(sqnorms, dim=1).mean(dim=1)
    return approx_tr_dzdx, N


def create_nn_time_independent(event_shape: Union[Tuple[int, ...], torch.Size],
                               hidden_size: int = 30,
                               n_hidden_layers: int = 2):
    event_size = int(torch.prod(torch.as_tensor(event_shape)))

    if hidden_size is None:
        hidden_size = max(4, int(3 * math.log(event_size)))
    hidden_shape = (hidden_size,)

    assert n_hidden_layers >= 0
    if n_hidden_layers == 0:
        layers = [diff_eq_layers.IgnoreLinear(event_shape, event_shape)]
    else:
        layers = [
            diff_eq_layers.IgnoreLinear(event_shape, hidden_shape),
            *[diff_eq_layers.IgnoreLinear(hidden_shape, hidden_shape) for _ in range(n_hidden_layers)],
            diff_eq_layers.IgnoreLinear(hidden_shape, event_shape)
        ]

    return TimeDerivativeDNN(layers)


def create_nn(event_shape: Union[Tuple[int, ...], torch.Size],
              hidden_size: int = None,
              n_hidden_layers: int = 2):
    event_size = int(torch.prod(torch.as_tensor(event_shape)))

    if hidden_size is None:
        hidden_size = max(4, int(3 * math.log(event_size)))
    hidden_shape = (hidden_size,)

    assert n_hidden_layers >= 0
    if n_hidden_layers == 0:
        layers = [diff_eq_layers.ConcatLinear(event_shape, event_shape)]
    else:
        layers = [
            diff_eq_layers.ConcatLinear(event_shape, hidden_shape),
            *[diff_eq_layers.ConcatLinear(hidden_shape, hidden_shape) for _ in range(n_hidden_layers)],
            diff_eq_layers.ConcatLinear(hidden_shape, event_shape)
        ]

    return TimeDerivativeDNN(layers)


def create_cnn(c: int, n_layers: int = 2):
    # c: number of image channels
    return TimeDerivativeDNN([ConcatConv2d(c, c) for _ in range(n_layers)])


def create_cnn_time_independent(c: int, n_layers: int = 2):
    # c: number of image channels
    return TimeDerivativeDNN([IgnoreConv2d(c, c) for _ in range(n_layers)])


class TimeDerivative(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x):
        # return dx/ dt
        raise NotImplementedError


class TimeDerivativeDNN(TimeDerivative):
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
        dx = x
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # Apply nonlinearity
            if i < len(self.layers) - 1:
                dx = torch.tanh(dx)
        assert dx.shape == x.shape
        return dx


class ODEFunction(nn.Module):
    def __init__(self, diffeq: callable):
        super().__init__()
        self.diffeq = diffeq
        self.register_buffer('_n_evals', torch.tensor(0.0))  # Counts the number of function evaluations

    def regularization(self):
        return torch.tensor(0.0)

    def before_odeint(self, **kwargs):
        self._n_evals.fill_(0)

    def forward(self, t, states):
        """

        :param t: shape ()
        :param states: (y0, y1, ..., yn) where yi.shape == (batch_size, event_size).
        :return:
        """
        raise NotImplementedError


class ExactODEFunction(ODEFunction):
    """
    Function that computes dx/dt with an exact log determinant of the Jacobian.
    """

    def __init__(self, diffeq: TimeDerivative):
        super().__init__(diffeq)

    def compute_log_det(self, t, x):
        raise NotImplementedError

    def forward(self, t, states):
        """

        :param t: shape ()
        :param states: (y0, y1, ..., yn) where yi.shape == (batch_size, event_size).
        :return:
        """
        assert len(states) >= 2
        y = states[0]
        self._n_evals += 1

        t = torch.as_tensor(t).type_as(y)

        with torch.enable_grad():
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])

        log_det = self.compute_log_det(t, y)
        assert torch.all(torch.isfinite(log_det))
        assert torch.all(~torch.isnan(log_det))
        return tuple([dy, log_det] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class ApproximateODEFunction(ODEFunction):
    """
    Function that computes dx/dt with an approximation for the log determinant of the Jacobian.
    """

    def __init__(self, network: TimeDerivativeDNN):
        super().__init__(diffeq=network)
        self.hutch_noise = None  # Noise tensor for Hutchinson trace estimation of the Jacobian

    def before_odeint(self, noise: torch.Tensor = None):
        super().before_odeint()
        self.hutch_noise = noise

    def forward(self, t, states):
        """

        :param t: shape ()
        :param states: (y0, y1, ..., yn) where yi.shape == (batch_size, event_size).
        :return:
        """
        assert len(states) >= 2
        y = states[0]
        self._n_evals += 1

        t = torch.as_tensor(t).type_as(y)

        if self.hutch_noise is None:
            self.hutch_noise = torch.randn_like(y)

        with torch.enable_grad():
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            divergence = self.divergence_step(dy, y)
        return tuple([dy, divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class RegularizedApproximateODEFunction(ApproximateODEFunction):
    def __init__(self,
                 network: TimeDerivativeDNN,
                 regularization: Union[str, Tuple[str, ...]] = ()):
        super().__init__(network)

        if isinstance(regularization, str):
            regularization = (regularization,)

        self.supported_reg_types = ['sq_jac_norm']
        for rt in regularization:
            if rt not in self.supported_reg_types:
                raise ValueError
        self.used_reg_types = regularization

        self.reg_jac_coef = 1.0
        self.stored_reg = None

    def divergence_step(self, dy, y) -> torch.Tensor:
        batch_size = y.shape[0]

        if "sq_jac_norm" in self.used_reg_types and self.training:
            divergence, sq_jac_norm = divergence_approx_extended(dy, y, e=self.hutch_noise)
            # Store regularization data
            self.stored_reg = self.reg_jac_coef * sq_jac_norm.mean()
        else:
            divergence = divergence_approx_basic(dy, y, e=self.hutch_noise)
        divergence = divergence.view(batch_size, 1)

        # TODO add other regularization terms

        return divergence

    def regularization(self):
        return (self.stored_reg or 0) + super().regularization()


class ContinuousBijection(Bijection):
    """
    Base class for bijections of continuous normalizing flows.

    Reference: Chen et al. "Neural Ordinary Differential Equations" (2019); https://arxiv.org/abs/1806.07366.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 f: ODEFunction,
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 end_time: float = 1.0,
                 solver: str = 'euler',  # Use euler (fastest solver)
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 **kwargs):
        """
        ContinuousBijection constructor.

        :param event_shape: shape of the event tensor.
        :param f: function to be integrated.
        :param context_shape: shape of the context tensor.
        :param end_time: integrate f from time 0 to this time. Default: 1.
        :param solver: which solver to use.
        :param atol: absolute tolerance for numerical integration.
        :param rtol: relative tolerance for numerical integration.
        :param kwargs: unused.
        """
        super().__init__(event_shape, context_shape)
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
        Inverse pass of the continuous bijection.

        :param z: tensor with shape `(*batch_shape, *event_shape)`.
        :param integration_times:
        :param kwargs: keyword arguments passed to self.f.before_odeint in the torchdiffeq solver.
        :return: transformed tensor and log determinant of the transformation.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        # Import from torchdiffeq locally, so the package does not break if torchdiffeq not installed
        from torchdiffeq import odeint

        # Flatten everything to facilitate computations
        batch_shape = get_batch_shape(z, self.event_shape)
        batch_size = int(torch.prod(torch.as_tensor(batch_shape)))
        z_flat = flatten_batch(flatten_event(z, self.event_shape), batch_shape)

        if integration_times is None:
            integration_times = self.make_integrations_times(z_flat)

        # Refresh odefunc statistics
        self.f.before_odeint(**kwargs)

        log_det_initial = torch.zeros(size=(batch_size, 1)).to(z_flat)
        state_t = odeint(
            self.f,
            (z_flat, log_det_initial),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_final_flat, log_det_final_flat = state_t[:2]

        # Reshape back to original shape
        x = unflatten_event(unflatten_batch(z_final_flat, batch_shape), self.event_shape)
        log_det = log_det_final_flat.view(*batch_shape)

        return x, log_det

    def forward(self,
                x: torch.Tensor,
                integration_times: torch.Tensor = None,
                noise: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the continuous bijection.

        :param torch.Tensor x: tensor with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor integration_times:
        :param torch.Tensor noise:
        :param kwargs: keyword arguments to be passed to `self.inverse`.
        :returns: transformed tensor and log determinant of the transformation.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if integration_times is None:
            integration_times = self.make_integrations_times(x)
        return self.inverse(
            x,
            integration_times=_flip(integration_times, 0),
            noise=noise,
            **kwargs
        )

    def regularization(self):
        return self.f.regularization()


class ExactContinuousBijection(ContinuousBijection):
    """
    Continuous NF bijection with an exact log Jacobian determinant.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], f: ExactODEFunction, **kwargs):
        super().__init__(event_shape, f, **kwargs)


class ApproximateContinuousBijection(ContinuousBijection):
    """
    Continuous NF bijection with an approximate log Jacobian determinant.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 f: ApproximateODEFunction,
                 **kwargs):
        """

        :param event_shape:
        :param f: function to be integrated.
        :param end_time: integrate f from t=0 to t=time_upper_bound. Default: 1.
        :param solver: which solver to use.
        :param kwargs:
        """
        super().__init__(event_shape, f, **kwargs)

    def make_integrations_times(self, z):
        return torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

    def inverse(self,
                z: torch.Tensor,
                integration_times: torch.Tensor = None,
                noise: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param z: tensor with shape (*batch_shape, *event_shape).
        :param integration_times:
        :param kwargs:
        :return:
        """
        # Import from torchdiffeq locally, so the package does not break if torchdiffeq not installed
        from torchdiffeq import odeint

        # Flatten everything to facilitate computations
        batch_shape = get_batch_shape(z, self.event_shape)
        batch_size = int(torch.prod(torch.as_tensor(batch_shape)))
        z_flat = flatten_batch(z, batch_shape)

        if integration_times is None:
            integration_times = self.make_integrations_times(z_flat)

        # Refresh odefunc statistics
        self.f.before_odeint(noise=noise)

        log_det_initial = torch.zeros(size=(batch_size, *([1] * len(self.event_shape)))).to(z_flat)
        state_t = odeint(
            self.f,
            (z_flat, log_det_initial),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_final_flat, log_det_final_flat = state_t[:2]

        # Reshape back to original shape
        x = unflatten_batch(z_final_flat, batch_shape)
        log_det = log_det_final_flat.view(*batch_shape)

        return x, log_det

    def forward(self,
                x: torch.Tensor,
                integration_times: torch.Tensor = None,
                noise: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if integration_times is None:
            integration_times = self.make_integrations_times(x)
        return self.inverse(
            x,
            integration_times=_flip(integration_times, 0),
            noise=noise,
            **kwargs
        )

    def regularization(self):
        return self.f.regularization()
