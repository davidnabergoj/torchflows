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
from typing import Union, Tuple

import torch
import torch.nn as nn

from torchflows.bijections.base import Bijection
from torchflows.bijections.continuous.time_derivative import TimeDerivative, ConcatConv2d, IgnoreConv2d, TimeDerivativeModule, TimeDerivativeModule, TimeDerivativeSequential
import torchflows.bijections.continuous.time_derivative as tderiv
from torchflows.bijections.continuous.util import approximate_divergence, delta_sq_norm_jac
from torchflows.utils import flatten_batch, get_batch_shape, unflatten_batch, unflatten_event


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class HutchinsonTimeDerivative(TimeDerivative):
    """Time derivative that approximes the divergence with the Hutchinson trace estimator.
    """

    def __init__(self,
                 forward_model: TimeDerivativeModule,
                 n_noise_samples: int = 1,
                 reg_jac: bool = False,
                 reg_jac_coef: float = 0.01,
                 reuse_noise: bool = False,
                 **kwargs):
        """HutchinsonTimeDerivative constructor.

        Note: the attribute _last_used_noise hold the last used noise samples for trace estimation.
        This can be useful when we want forward and inverse CNF maps to stay identical, e.g., in reconstruction accuracy 
            testing or exact target-latent transformations.

        :param TimeDerivativeModule forward_model: module that receives as input a time tensor with shape `()` and space tensor 
            with shape `(batch_size, event_size)`, and outputs a time derivative of the space tensor with shape 
            `(batch_size, event_size)`.
        :param int n_noise_samples: number of noise samples for trace estimation.
        :param bool reg_jac: if True, use Jacobian regularization.
        :param float reg_jac_coef: jacobian regularization coefficient.
        :param bool reuse_noise: if True, reuse the same Hutchinson noise vectors in every iteration.
        :param kwargs: keyword arguments for TimeDerivative.
        """
        super().__init__(**kwargs)
        self.forward_model = forward_model
        self.n_noise_samples = n_noise_samples

        # Regularization
        self.reg_jac = reg_jac
        self.reg_jac_coef = reg_jac_coef

        # For deterministic forward and inverse passes
        self.reuse_noise = reuse_noise
        self._reusable_noise: torch.Tensor = None  # Hutchinson noise samples with shape 
        # `(n_noise_samples, batch_size, event_size)` for divergence estimation.

    @property
    def reg_jac_active(self) -> bool:
        """Return True if we are currently using Jacobian regularization."""
        return self.training and self.reg_jac and self.reg_jac_coef > 0

    def prepare_initial_state(self,
                              z0: torch.Tensor):
        """Prepare the initial state.

        :param torch.Tensor z0: event tensor with shape `(batch_size, event_size)`.
        :rtype Tuple[torch.Tensor, ...].
        """
        div0 = torch.zeros(
            size=(z0.shape[0],), 
            dtype=z0.dtype, 
            device=z0.device
        )
        if self.reg_jac_active:
            return (z0, div0, div0.clone())
        else:
            return (z0, div0)

    def step(self,
             t: torch.Tensor,
             x: torch.Tensor,
             sq_norm_jac: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """Compute dx/dt and the corresponding divergence.

        :param torch.Tensor t: time tensor with shape `()`.
        :param torch.Tensor x: spatial tensor with shape `(batch_size, *event_shape)`.
        :param Optional[torch.Tensor] sq_norm_jac: delta tensor for the squared norm of the Jacobian with shape 
            (`batch_size`,).
        :rtype: Tuple[torch.Tensor, ...].
        :return: dx/dt tensor with shape `(batch_size, *event_shape)`, divergence tensor with shape `(batch_size,)`, and 
            possible Jacobian delta tensor with shape `(batch_size,)`
        """
        dxdt = self.forward_model(t, x)

        if dxdt.shape != x.shape:
            raise ValueError(f"Expected dxdt and x to have equal shapes, but got {dxdt.shape = }, {x.shape = }")

        if self.reuse_noise:
            if self._reusable_noise is None or self._reusable_noise.shape[1:] != x.shape:
                self._reusable_noise = torch.randn(size=(self.n_noise_samples, *x.shape))
            noise = self._reusable_noise
        else:
            noise = torch.randn(size=(self.n_noise_samples, *x.shape))

        noise_tuple = tuple([e for e in noise])

        if self.reg_jac_active:
            if sq_norm_jac is None:
                raise ValueError("Missing integrated squared norm of the Jacobian.")
            app_div, e_dzdx = approximate_divergence(
                dz=dxdt,
                x=x,
                noise=noise_tuple,
                return_e_dzdx=True
            )
            delta_jac = delta_sq_norm_jac(e_dzdx=e_dzdx).view_as(sq_norm_jac)
            return dxdt, app_div, sq_norm_jac + delta_jac
        else:
            app_div = approximate_divergence(
                dz=dxdt,
                x=x,
                noise=noise_tuple
            )
            return dxdt, app_div


def create_nn_time_independent(event_shape: Union[Tuple[int, ...], torch.Size],
                               hidden_size: int = 30,
                               n_hidden_layers: int = 2):
    event_size = int(torch.prod(torch.as_tensor(event_shape)))

    if hidden_size is None:
        hidden_size = max(4, int(3 * math.log(event_size)))
    hidden_shape = (hidden_size,)

    assert n_hidden_layers >= 0
    if n_hidden_layers == 0:
        layers = [tderiv.IgnoreLinear(event_shape, event_shape)]
    else:
        layers = [
            tderiv.IgnoreLinear(event_shape, hidden_shape),
            *[tderiv.IgnoreLinear(hidden_shape, hidden_shape)
              for _ in range(n_hidden_layers)],
            tderiv.IgnoreLinear(hidden_shape, event_shape)
        ]

    return TimeDerivativeSequential(layers)


class TimeTanh(nn.Tanh):
    """Tanh subclass that ignores the time component."""

    def forward(self, t, x):
        """Apply tanh to x.

        :param torch.Tensor t: unused.
        :param torch.Tensor x: tensor with shape `(batch_size, event_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, event_size)`.
        """
        return super().forward(x)


def create_dnn_forward_model(event_shape: Union[Tuple[int, ...], torch.Size],
                             hidden_size: int = None,
                             n_hidden_layers: int = 2) -> TimeDerivativeSequential:
    """Create time derivative neural network with linear layers.
    The time and space components are concatenated.
    """
    if n_hidden_layers < 0:
        raise ValueError("Number of hidden layers must be non-negative.")

    event_size = int(torch.prod(torch.as_tensor(event_shape)))

    if hidden_size is None:
        hidden_size = max(4, int(3 * math.log(event_size)))
    hidden_shape = (hidden_size,)

    if n_hidden_layers == 0:
        layers = [tderiv.ConcatLinear(event_shape, event_shape)]
    else:
        layers = [
            tderiv.ConcatLinear(event_shape, hidden_shape),
            TimeTanh()
        ]
        for _ in range(n_hidden_layers):
            layers.extend([
                tderiv.ConcatLinear(hidden_shape, hidden_shape),
                TimeTanh()
            ])
        layers.append(tderiv.ConcatLinear(hidden_shape, event_shape))

    return TimeDerivativeSequential(layers)


def create_cnn(c: int, n_layers: int = 2):
    # c: number of image channels
    return TimeDerivativeSequential([ConcatConv2d(c, c) for _ in range(n_layers)])


def create_cnn_time_independent(c: int, n_layers: int = 2):
    # c: number of image channels
    return TimeDerivativeSequential([IgnoreConv2d(c, c) for _ in range(n_layers)])


class ContinuousBijection(Bijection):
    """
    Base class for bijections of continuous normalizing flows.

    Reference: Chen et al. "Neural Ordinary Differential Equations" (2019); https://arxiv.org/abs/1806.07366.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 f: TimeDerivative,
                 solver: str = 'rk4',
                 context_shape: Union[torch.Size, Tuple[int, ...]] = None,
                 end_time: float = 1.0,
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 **kwargs):
        """
        ContinuousBijection constructor.

        :param event_shape: shape of the event tensor.
        :param f: time derivative function to be integrated.
        :param context_shape: shape of the context tensor.
        :param end_time: integrate f from time 0 to this time. Default: 1.
        :param solver: which solver to use.
        :param atol: absolute tolerance for numerical integration.
        :param rtol: relative tolerance for numerical integration.
        :param kwargs: unused.
        """
        super().__init__(event_shape, context_shape)
        self.f = f
        self.register_buffer(
            "sqrt_end_time",
            torch.sqrt(torch.tensor(end_time))
        )
        self.end_time = end_time
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def make_integrations_times(self, z):
        return torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

    def inverse(self,
                z: torch.Tensor,
                integration_times: torch.Tensor = None,
                before_odeint_kwargs: dict = None,
                return_aux: bool = False,
                context: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Inverse pass of the continuous bijection.

        :param z: tensor with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor integration_times: tensor of integration times. This is torch.tensor([0, 1]) by default.
        :param before_odeint_kwargs: keyword arguments passed to self.f.before_odeint in the torchdiffeq solver.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :return: transformed tensor, log determinant of the transformation, and possible auxiliary tensors.
        """
        before_odeint_kwargs = before_odeint_kwargs or {}

        # Import from torchdiffeq locally, so the rest of the package does not break if torchdiffeq not installed
        try:
            from torchdiffeq import odeint
        except ImportError as e:
            raise e

        # Flatten batch to facilitate computations
        batch_shape = get_batch_shape(z, self.event_shape)
        z0 = flatten_batch(z, batch_shape)

        if integration_times is None:
            integration_times = self.make_integrations_times(z0)

        # Refresh odefunc statistics
        self.f.before_odeint(**kwargs)

        state_0 = self.f.prepare_initial_state(z0=z0)
        state_t = odeint(
            func=self.f,
            y0=state_0,
            t=integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            **kwargs
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        zT, divT, *aux = state_t

        # Reshape back to original shape
        x = unflatten_batch(zT, batch_shape)
        log_det = divT.view(*batch_shape)

        if return_aux:
            return x, log_det, *aux
        else:
            return x, log_det

    def forward(self,
                x: torch.Tensor,
                integration_times: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the continuous bijection.

        :param torch.Tensor x: tensor with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor integration_times:
        :param kwargs: keyword arguments to be passed to `self.inverse`.
        :returns: transformed tensor and log determinant of the transformation.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if integration_times is None:
            integration_times = self.make_integrations_times(x)
        return self.inverse(
            z=x,
            integration_times=_flip(integration_times, 0),
            **kwargs
        )

    def regularization(self,
                       *aux: Tuple[torch.Tensor, ...]):
        """Compute regularization.

        :param Tuple[torch.Tensor, ...] aux: regularization terms, each with shape `()`.
        :rtype: Union[torch.Tensor].
        :return: tensor with shape `()`.
        """
        if len(aux) == 0:
            return torch.tensor(0.0)
        else:
            raise NotImplementedError


    def sample(self,
               sample_shape: Union[int, torch.Size, Tuple[int, ...]],
               context: torch.Tensor = None,
               no_grad: bool = False,
               return_log_prob: bool = False,
               return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        if context is not None:
            z = self.base_sample(sample_shape=sample_shape)
            if get_batch_shape(context, self.context_shape) == sample_shape:
                # Option A: a context tensor is given for each sampled element
                pass
            else:
                # Option B: one context tensor is given for the entire to-be-sampled batch
                sample_shape = (*sample_shape, len(context))
                context = context[None].repeat(
                    *[*sample_shape, *([1] * len(context.shape))])  # Make context shape match z shape
                assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base_sample(sample_shape=sample_shape)

        if no_grad:
            z = z.detach()
            with torch.no_grad():
                x, log_det, *aux = self.bijection.inverse(
                    z.view(*sample_shape, *self.bijection.event_shape),
                    context=context,
                    return_aux=return_aux
                )
        else:
            x, log_det, *aux = self.bijection.inverse(
                z.view(*sample_shape, *self.bijection.event_shape),
                context=context,
                return_aux=return_aux
            )
        x = x.to(self.get_device())

        if return_log_prob:
            log_prob = self.base_log_prob(z) + log_det
            return x, log_prob, *aux
        return x, *aux
