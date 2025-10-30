import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchflows.bijections.continuous.base import TimeDerivative, ContinuousBijection
from torchflows.bijections.continuous.time_derivative import TimeDerivativeModule
from torchflows.utils import flatten_event

def concatenate_x_t(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Concatenate `t` to the end of `x`.
    
    :param torch.Tensor x: event (space) tensor with shape `(batch_size, event_size)`.
    :param torch.Tensor t: time tensor with shape `()`.
    :rtype: torch.Tensor.
    :return: tensor with shape `(batch_size, event_size + 1)` where `t` is concatenated to the end of dimension 1.
    """
    if len(x.shape) != 2:
        raise ValueError(f"Shape of x must be (batch_size, event_size), but got {x.shape = }")
    if t.shape != ():
        raise ValueError(f"Shape of t must be (), but got {t.shape = }")
    s = torch.nn.functional.pad(torch.clone(x), pad=(0, 1), value=1.0)
    s[..., -1] = t
    return s


class OTResNet(nn.Module):
    """Two-layer ResNet as described in the original OTFlow paper.
    """

    def __init__(self, c_event_size: int, hidden_size: int, step_size: float):
        """OTResNet constructor.

        :param int c_event_size: size of the event tensor with the concatenated 
         time dimension.
        :param int hidden_size: number of hidden dimensions "m".
        :param float step_size: "h" in the original formulation.
        """
        super().__init__()

        self.c_event_size = c_event_size
        d = math.sqrt(hidden_size)

        self.K0 = nn.Parameter(torch.randn(hidden_size, self.c_event_size) / d)
        self.K1 = nn.Parameter(torch.randn(hidden_size, hidden_size) / d)

        self.b0 = nn.Parameter(torch.randn(hidden_size,) / d)
        self.b1 = nn.Parameter(torch.randn(hidden_size,) / d)

        self.step_size = step_size

    @staticmethod
    def sigma(x):
        return torch.logaddexp(x, -x)

    @staticmethod
    def sigma_prime(x):
        return torch.tanh(x)

    @staticmethod
    def sigma_prime2(x):
        return 1 - torch.tanh(x) ** 2

    def compute_u0(self, s):
        """Compute u0 from Equation 11.

        :param torch.Tensor s: tensor with shape `(batch_size, c_event_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, hidden_size)`.
        """
        lin = torch.nn.functional.linear(s, self.K0, self.b0)
        return self.sigma(lin)

    def compute_u1(self, u0):
        """Compute u1 from Equation 11.

        :param torch.Tensor u0: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, hidden_size)`.
        """
        lin = torch.nn.functional.linear(u0, self.K1, self.b1)
        return u0 + self.step_size * self.sigma(lin)

    def forward(self, s: torch.Tensor):
        """Compute N(s; theta_N) from Equation 11.

        :param torch.Tensor s: tensor with shape `(batch_size, c_event_size)`.
        :param torch.Tensor w: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, c_event_size)`.
        """
        return self.compute_u1(self.compute_u0(s=s))

    def compute_z1(self, w: torch.Tensor, u0: torch.Tensor):
        """Compute z1 from Equation 13.

        :param torch.Tensor w: tensor with shape `(batch_size, hidden_size)`.
        :param torch.Tensor u0: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, hidden_size)`.
        """
        lin = torch.nn.functional.linear(u0, self.K1, self.b1)
        act = self.sigma_prime(lin * w)
        lin2 = torch.nn.functional.linear(act, self.K1.T)
        return w + self.step_size * lin2

    def compute_z0(self, s: torch.Tensor, z1: torch.Tensor):
        """Compute z0 from Equation 13.

        :param torch.Tensor s: tensor with shape `(batch_size, c_event_size)`.
        :param torch.Tensor z1: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, c_event_size)`.
        """
        lin = torch.nn.functional.linear(s, self.K0, self.b0)
        act = self.sigma_prime(lin)
        return torch.nn.functional.linear(act * z1, self.K0.T)

    def jvp(self, s: torch.Tensor, w: torch.Tensor):
        """Compute grad_ResNet_wrt_s * w. The first term is the jacobian matrix.

        :param torch.Tensor s: tensor with shape `(batch_size, c_event_size)`.
        :param torch.Tensor w: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, c_event_size)`.
        """
        u0 = self.compute_u0(s=s)
        z1 = self.compute_z1(w=w, u0=u0)
        z0 = self.compute_z0(s=s, z1=z1)
        return z0

    def hessian_trace(self,
                      s: torch.Tensor,
                      w: torch.Tensor,
                      u0: torch.Tensor,
                      z1: torch.Tensor):
        """Compute OTResNet Hessian trace from Equation 15.

        :param torch.Tensor s: tensor with shape `(b, e)`.
        :param torch.Tensor w: tensor with shape `(b, h)`.
        :param torch.Tensor u0: tensor with shape `(b, h)`.
        :param torch.Tensor z1: tensor with shape `(b, h)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(b,)`.
        """
        lin_t0 = torch.nn.functional.linear(s, self.K0, self.b0)  # (b, h)
        t0_a = self.sigma_prime2(lin_t0) * z1  # (b, h)
        K0_E = self.K0[:, :-1]  # (h, e - 1)
        t0_b = (K0_E * K0_E).sum(dim=1)  # (h,)

        t0 = torch.sum(t0_a * t0_b[None], dim=1)  # (b,)

        lin_t1_a = torch.nn.functional.linear(u0, self.K1, self.b1)  # (b, h)
        t1_a = self.sigma_prime2(lin_t1_a) * w  # (b, h)
        J = torch.multiply(
            self.sigma_prime(
                torch.nn.functional.linear(s, self.K0, self.b0)
            )[:, :, None],  # (b, h, 1)
            K0_E[None, :, :]  # (1, h, e - 1)
        )  # (b, h, e - 1)
        K1J = torch.matmul(  # Multiplies along the batch
            self.K1,  # (h, h)
            J  # (b, h, e - 1)
        )  # (b, h, e - 1)
        t1_b = (K1J * K1J).sum(dim=-1)  # (b, h)

        t1 = torch.sum(t1_a * t1_b, dim=1)  # (b,)

        return t0 + self.step_size * t1


class OTPotential(TimeDerivativeModule):    
    """OT-Flow potential for general tensors."""
    def __init__(self,
                 event_size: int,
                 hidden_size: int = None,
                 step_size: float = 1.0):
        """OT-Flow potential constructor.
        Uses the same initialization as the implementation from the original 
         paper.
        Reference: https://github.com/EmoryMLIP/OT-Flow/blob/master/src/Phi.py.

        :param int event_size: size of the event tensor without the concatenated 
         time dimension.
        :param int hidden_size: number of neurons in the hidden ResNet layer.
        :param float step_size: ResNet step size.
        """
        super().__init__()

        # hidden_size = m
        if hidden_size is None:
            hidden_size = max(3 * int(math.log(event_size)), 4)

        r = min(10, event_size)

        self.w = nn.Parameter(torch.ones(hidden_size))
        self.A = nn.Parameter(torch.randn(r, event_size + 1))
        self.A = nn.init.xavier_uniform_(self.A)  # overwrite initialization
        self.b = nn.Parameter(torch.zeros(event_size + 1))

        self.resnet = OTResNet(
            c_event_size=event_size + 1,
            hidden_size=hidden_size,
            step_size=step_size
        )

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the potential value.
        
        :param torch.Tensor t: time tensor with shape `()`.
        :param torch.Tensor x: event (space) tensor with shape `(batch_size, event_size)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor].
        :return: space derivative of the potential with shape `(batch_size, event_size)` and
            time derivative of the potential with shape `(batch_size, 1)`.
        """
        return self.gradient(concatenate_x_t(x, t))

    def gradient(self, s: torch.Tensor):
        """
        Compute the gradient of the OT potential w.r.t. the concatenated state.

        :param torch.Tensor s: tensor with shape `(batch_size, event_size + 1)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor].
        :return: space derivative with shape `(batch_size, event_size)` and time derivative with shape 
            `(batch_size, 1)`.
        """
        batch_size = s.shape[0]

        # Equation 12
        lin = torch.nn.functional.linear(s, self.A.T @ self.A, self.b)
        grad = self.resnet.jvp(s, self.w) + lin
        space_derivative = grad[..., :-1].view(batch_size, -1)
        time_derivative = grad[..., -1].view(batch_size, 1)
        return space_derivative, time_derivative

    def compute_divergence(self,
                           s: torch.Tensor,
                           u0: torch.Tensor,
                           z1: torch.Tensor):
        """Compute divergence.

        :param torch.Tensor s: tensor with shape `(batch_size, event_size + 1)`.
        :param torch.Tensor u0: tensor with shape `(batch_size, hidden_size)`.
        :param torch.Tensor z1: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size,)`.
        """
        # Equation 14
        tr_first_term = self.resnet.hessian_trace(s, self.w, u0, z1)

        # Second term: tr(E^T (A^T A) E)
        # E.T @ A ... remove last row (assuming E has d of d+1 standard basis vectors)
        # A @ E ... remove last column (assuming E has d of d+1 standard basis vectors)
        tr_second_term = torch.trace((self.A.T @ self.A)[:-1, :-1])

        return -(tr_first_term + tr_second_term)


class OTFlowTimeDerivative(TimeDerivative):
    """OT-Flow time derivative, as described in Appendix D, Equation 26.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 reg_transport: bool = True,
                 reg_transport_coef: float = 0.01,
                 reg_hjb: bool = True,
                 reg_hjb_coef: float = 0.01,
                 **kwargs):
        """OTFlowTimeDerivative constructor.

        :param event_shape: shape of the event (space) tensor.
        :param bool reg_transport: if True, use transport cost regularization.
        :param float reg_transport_cef: transport cost regularization coefficient.
        :param bool reg_hjb: if True, use HJB regularization.
        :param float reg_hjb_coef: HJB regularization coefficient.
        :param kwargs: keyword arguments for OTPotential.
        """
        super().__init__()
        event_size = int(torch.prod(torch.as_tensor(event_shape)))

        self.time_deriv = OTPotential(event_size=event_size, **kwargs)
        self.reg_transport = reg_transport
        self.reg_transport_coef = reg_transport_coef
        self.reg_hjb = reg_hjb
        self.reg_hjb_coef = reg_hjb_coef

    @property
    def reg_transport_active(self):
        return self.training and self.reg_transport and self.reg_transport_coef > 0
    
    @property
    def reg_hjb_active(self):
        self.training and self.reg_hjb and self.reg_hjb_coef > 0

    @torch.no_grad()
    def prepare_initial_state(self, z0: torch.Tensor):
        div0 = torch.zeros(
            size=(z0.shape[0],),
            dtype=z0.dtype, 
            device=z0.device
        )

        if self.reg_transport_active and self.reg_hjb_active:
            return (z0, div0, div0.clone(), div0.clone())
        elif self.reg_transport_active and not self.reg_hjb_active:
            return (z0, div0, div0.clone())
        elif not self.reg_transport_active and self.reg_hjb_active:
            return (z0, div0, div0.clone())
        else:
            return (z0, div0)

    def step(self,
             t: torch.Tensor,
             x: torch.Tensor,
             *aux: Tuple[torch.Tensor, ...]):
        """Compute dx/dt, divergence, transport delta, and HJB delta.

        This means computing partial_t [z(x, t), ell(x, t), L(x, t), R(x, t)] where
            z(x, t) = -grad phi (z(x, t), t; theta)
            ell(x, t) = -tr(hess phi (z(x, t), t; theta))
            L(x, t) = 1/2 || grad phi(z(x, t), t; theta) ||^2
            R(x, t) = | partial_t phi(z(x, t), t; theta) - 1/2 || grad phi (z(x, t), t; theta)||^2 |
        Refer to Section 3 for equations.

        :param torch.Tensor t: time tensor with shape `()`.
        :param torch.Tensor x: space tensor with shape `(batch_size, *event_shape)`.
        :param Tuple[torch.Tensor, ...] aux: unused.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]].
        :return: dx/dt with the same shape as x and divergence tensor with shape 
         `(batch_size, 1)`. If training, also return the delta transport cost and the delta HJB.
        """

        x_flat = flatten_event(x, event_shape=x.shape[1:])

        # Work in flattened space

        self._n_evals += 1
        s = concatenate_x_t(x_flat, t)

        # Gradient computation
        grad_space, grad_time = self.time_deriv.gradient(s=s)

        u0 = self.time_deriv.resnet.compute_u0(s=s)
        z1 = self.time_deriv.resnet.compute_z1(w=self.time_deriv.w, u0=u0)
        div = self.time_deriv.compute_divergence(s=s, u0=u0, z1=z1)

        dxdt = (-grad_space).view_as(x)  # Reshape into original space

        if self.reg_transport_active and not self.reg_hjb_active:
            d_transport = 1/2 * torch.sum(grad_space ** 2, dim=-1)
            return (dxdt, div, d_transport)
        elif self.reg_transport_active and self.reg_hjb_active:
            d_transport = 1/2 * torch.sum(grad_space ** 2, dim=-1)
            d_hjb = torch.abs(grad_time - d_transport)
            return (dxdt, div, d_transport, d_hjb)
        elif not self.reg_transport_active and self.reg_hjb_active:
            d_transport = 1/2 * torch.sum(grad_space ** 2, dim=-1)
            d_hjb = torch.abs(grad_time - d_transport)
            return (dxdt, div, d_hjb)
        elif not self.reg_transport_active and not self.reg_hjb_active:
            return (dxdt, div)
        else:
            raise RuntimeError


class OTFlowBijection(ContinuousBijection):
    """OT-flow architecture for general tensors.
    Parameterizes the time derivative with a non-convolutional ResNet.

    Onken et al. "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport" (2021).
    URL: https://arxiv.org/abs/2006.00104.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """OTFlowBijection constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param dict nn_kwargs: keyword arguments for `create_nn`.
        :param dict time_derivative_kwargs: keyword arguments for `ApproximateTimeDerivative`.
        :param kwargs: keyword arguments for `ContinuousBijection`.
        """
        super().__init__(
            event_shape=event_shape,
            f=OTFlowTimeDerivative(
                event_shape=event_shape,
                **(time_derivative_kwargs or {})
            ),
            **kwargs
        )

    def regularization(self, *aux: torch.Tensor) -> torch.Tensor:
        """Compute OT-Flow regularization.
        
        :param Tuple[torch.Tensor, ...] aux: possible transport and HJB cost tensors. If provided, both tensors have 
            shape `(batch_size,)`.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`.
        """
        if self.f.reg_transport and self.f.reg_hjb:
            transport_cost, hjb_cost = aux
        elif self.f.reg_transport and not self.f.reg_hjb:
            transport_cost = aux[0]
            hjb_cost = torch.tensor(0.0)
        elif not self.f.reg_transport and self.f.reg_hjb:
            transport_cost = torch.tensor(0.0)
            hjb_cost = aux[0]
        else:
            return torch.tensor(0.0)
        
        return self.f.reg_transport_coef * transport_cost.mean() + self.f.reg_hjb_coef * hjb_cost.mean()