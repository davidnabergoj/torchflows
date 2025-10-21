import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchflows.bijections.continuous.base import TimeDerivative, ContinuousBijection, ODEFunction
from torchflows.utils import flatten_event, flatten_batch, get_batch_shape, unflatten_batch, unflatten_event

def concatenate_x_t(x, t):
    # Concatenate t to the end of x
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
        :return: tensor with shape `(batch_size, hidden_size)`.
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


class OTPotential(TimeDerivative):
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

    def forward(self, t, x):
        return self.gradient(
            concatenate_x_t(x, t),
        )

    def gradient(self, s: torch.Tensor):
        """
        Compute the gradient of the OT potential w.r.t. the concatenated state.

        :param torch.Tensor s: tensor with shape `(batch_size, event_size + 1)`.
        :rtype: Tuple[torch.Tensor, torch.Tensor].
        :return: tuple of tensors. 
         The first element is the space derivative with shape 
          `(batch_size, event_size)`. 
         The second element is the time derivative with shape `(batch_size,)`.
        """
        # Equation 12
        lin = torch.nn.functional.linear(s, self.A.T @ self.A, self.b)
        grad = self.resnet.jvp(s, self.w) + lin
        space_derivative = grad[..., :-1]
        time_derivative = grad[..., -1]
        return space_derivative, time_derivative

    def hessian_trace(self, 
                      s: torch.Tensor, 
                      u0: torch.Tensor, 
                      z1: torch.Tensor):
        """Compute trace of the Hessian of the potential.
        
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

        return tr_first_term + tr_second_term


class OTFlowODEFunction(ODEFunction):
    """OT-Flow ODE, as described in Appendix D, Equation 26.
    """
    def __init__(self, n_dim, **kwargs):
        """OTFlowODEFunction constructor.
        """
        super().__init__(OTPotential(n_dim, **kwargs))

    def forward(self, 
                t: float, 
                states: Tuple[torch.Tensor, ...]):
        """Compute partial_t [z(x, t), ell(x, t), L(x, t), R(x, t)] where
            z(x, t) = -grad phi (z(x, t), t; theta)
            ell(x, t) = -tr(hess phi (z(x, t), t; theta))
            L(x, t) = 1/2 || grad phi(z(x, t), t; theta) ||^2
            R(x, t) = | partial_t phi(z(x, t), t; theta) - 1/2 || grad phi (z(x, t), t; theta)||^2 |
        Refer to Section 3 for equations.

        :param float t: time parameter t.
        :param Tuple[torch.Tensor, ...] states: tuple of tensors 
         (z(x, t), ell(x, t), L(x, t), R(x, t)).
         z(x, t) has shape `(batch_size, n_dim + 1)`.
        :rtype: Tuple[torch.Tensor, ....].
        :return: tuple of tensors 
            (d_z(x, t), d_ell(x, t), d_L(x, t), d_R(x, t)).
        """
        self._n_evals += 1
        x, _, _, _ = states[:4]  # (s, ell, L, R).
        t = torch.as_tensor(t, dtype=x.dtype)

        s = concatenate_x_t(x, t)

        # Gradient computation
        grad_space, grad_time = self.diffeq.gradient(s=s)

        u0 = self.diffeq.resnet.compute_u0(s=s)
        z1 = self.diffeq.resnet.compute_z1(w=self.diffeq.w, u0=u0)
        tr_hess = self.diffeq.hessian_trace(s=s, u0=u0, z1=z1)

        d_ell = 1/2 * torch.sum(grad_space ** 2, dim=-1)
        return tuple([
            -grad_space,
            -tr_hess,
            d_ell,
            torch.abs(grad_time - d_ell)
        ])

    def compute_log_det(self, t, x):
        s = concatenate_x_t(x, t)
        self.diffeq: OTPotential

        w = self.diffeq.w
        u0 = self.diffeq.resnet.compute_u0(s=s)
        z1 = self.diffeq.resnet.compute_z1(w=w, u0=u0)

        return -self.diffeq.hessian_trace(
            s=s,
            u0=u0,
            z1=z1
        ).view(-1, 1)  # Need an empty dim at the end


class OTFlow(ContinuousBijection):
    """Optimal transport flow (OT-flow) architecture.

    Reference: Onken et al. "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport" (2021); https://arxiv.org/abs/2006.00104.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 ode_kwargs: dict = None,
                 solver: str = 'rk4',
                 alpha1: float = 1.0,
                 alpha2: float = 1.0,
                 alpha3: float = 1.0,
                 **kwargs):
        """OTFlow constructor.

        :param event_shape: shape of the event tensor.
        :param dict ode_kwargs: keyword arguments for OTFlowODEFunction.
        :param str solver: ODE solver name. Default: 'rk4'.
        :param float alpha1: loss coefficient for the terminal constraint.
        :param float alpha2: loss coefficient for the kinetic energy (i.e., 
         flow smoothness).
        :param float alpha3: loss coefficient for regularization terms.
        :param kwargs: keyword arguments for ExactContinuousBijection.
        """
        ode_kwargs = ode_kwargs or {}

        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = OTFlowODEFunction(n_dim, **ode_kwargs)
        super().__init__(event_shape, diff_eq, solver=solver, **kwargs)

    def inverse(self,
                z: torch.Tensor,
                integration_times: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the OT-Flow bijection.
        Integrates Equation 26 from time integration_times[0] to time integration_times[1].

        :param torch.Tensor z: tensor with shape `(*batch_shape, *event_shape)`.
        :param torch.Tensor integration_times: float tensor with two elements
         that determine the start and end integration time.
        :param kwargs: keyword arguments passed to self.f.before_odeint in the torchdiffeq solver.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        :return: tuple of four tensors - integrated state, log determinant, 
         transport cost, and HJB regularizer.
        """
        from torchdiffeq import odeint
        
        # Flatten everything to facilitate computations
        batch_shape = get_batch_shape(z, self.event_shape)
        batch_size = int(torch.prod(torch.as_tensor(batch_shape)))
        z_flat_0 = flatten_batch(
            flatten_event(z, self.event_shape), 
            batch_shape
        )

        if integration_times is None:
            integration_times = self.make_integrations_times(z_flat_0)

        # Refresh odefunc statistics
        self.f.before_odeint(**kwargs)

        log_det_0 = torch.zeros(
            size=(batch_size, 1),
            device=z_flat_0.device,
            dtype=z_flat_0.dtype
        )
        transport_cost_0 = torch.zeros_like(log_det_0)
        hjb_0 = torch.zeros_like(log_det_0)

        ode_state_initial = (z_flat_0, log_det_0, transport_cost_0, hjb_0)
        ode_state_t = odeint(
            self.f,
            ode_state_initial,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            **kwargs
        )

        if len(integration_times) == 2:
            ode_state_t = tuple(s[1] for s in ode_state_t)

        z_flat_T, log_det_T, transport_cost_T, hjb_T = ode_state_t

        # Reshape back to original shape
        x = unflatten_event(
            unflatten_batch(z_flat_T, batch_shape), 
            self.event_shape
        )
        log_det = log_det_T.view(*batch_shape)
        transport_cost = transport_cost_T.view(*batch_shape)
        hjb = hjb_T.view(*batch_shape)

        return x, log_det, transport_cost, hjb
    
