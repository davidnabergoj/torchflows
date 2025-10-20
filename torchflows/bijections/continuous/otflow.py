import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchflows.bijections.continuous.base import ExactODEFunction, TimeDerivative, ExactContinuousBijection


def concatenate_x_t(x, t):
    # Concatenate t to the end of x
    s = torch.nn.functional.pad(torch.clone(x), pad=(0, 1), value=1.0)
    s[..., -1] = t
    return s

class OTNetwork(nn.Module):
    """
    
    """
    def __init__(self, event_size: int):
        self.event_size = event_size
        super().__init__()

class OTResNet(OTNetwork):
    """Two-layer ResNet as described in the original OTFlow paper.
    """

    def __init__(self, event_size: int, hidden_size: int, step_size: float = 1.0):
        """OTResNet constructor.

        :param hidden_size: number of hidden dimensions "m".
        :param step_size: "h" in the original formulation.
        """
        super().__init__(event_size)

        self.K0 = nn.Parameter(torch.randn(size=(hidden_size, self.event_size)))
        self.K1 = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)))

        self.b0 = nn.Parameter(torch.randn(size=(hidden_size,)))
        self.b1 = nn.Parameter(torch.randn(size=(hidden_size,)))

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

        :param torch.Tensor s: tensor with shape `(batch_size, event_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, hidden_size)`.
        """
        return self.sigma(torch.nn.functional.linear(s, self.K0, self.b0))

    def compute_u1(self, u0):
        """Compute u1 from Equation 11.

        :param torch.Tensor u0: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, hidden_size)`.
        """
        return u0 + self.step_size * self.sigma(torch.nn.functional.linear(u0, self.K1, self.b1))

    def forward(self, s: torch.Tensor):
        """Compute N(s; theta_N) from Equation 11.

        :param torch.Tensor s: tensor with shape `(batch_size, event_size)`.
        :param torch.Tensor w: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, event_size)`.
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

        :param torch.Tensor s: tensor with shape `(batch_size, event_size)`.
        :param torch.Tensor z1: tensor with shape `(batch_size, hidden_size)`.
        :rtype: torch.Tensor.
        :return: tensor with shape `(batch_size, event_size)`.
        """
        lin = torch.nn.functional.linear(s, self.K0, self.b0)
        act = self.sigma_prime(lin)
        return torch.nn.functional.linear(act * z1, self.K0.T)
    
    def jvp(self, s: torch.Tensor, w: torch.Tensor):
        """Compute grad_ResNet_wrt_s * w. The first term is the jacobian matrix.

        :param torch.Tensor s: tensor with shape `(batch_size, event_size)`.
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
        K1J = torch.swapdims(
            torch.matmul(
                self.K1,  # (h, h)
                torch.swapdims(J, 0, 1)  # (h, b, e - 1)
            ),  # (h, b, e - 1)
            0, 
            1
        )  # (b, h, e - 1)
        t1_b = (K1J * K1J).sum(dim=-1)  # (b, h)

        t1 = torch.sum(t1_a * t1_b, dim=1)  # (b,)

        return t0 + self.step_size * t1


class OTPotential(TimeDerivative):
    def __init__(self, event_size: int, hidden_size: int = None, resnet_kwargs: dict = None):
        super().__init__()
        resnet_kwargs = resnet_kwargs or {}

        # hidden_size = m
        if hidden_size is None:
            hidden_size = max(3 * int(math.log(event_size)), 4)

        r = min(10, event_size)

        # Initialize w to 1
        # Initialize A to identity
        # Initialize b to 0

        divisor = event_size ** 2

        delta_w = torch.randn(size=(hidden_size,)) / divisor
        delta_A = torch.randn(size=(r, event_size + 1)) / divisor
        delta_b = torch.randn(size=(event_size + 1,)) / divisor

        self.w = nn.Parameter(1 + delta_w)
        self.A = nn.Parameter(torch.eye(r, event_size + 1) + delta_A)
        self.b = nn.Parameter(0 + delta_b)
        self.resnet = OTResNet(event_size + 1, hidden_size, **resnet_kwargs)  # (x, t) has d+1 elements

    def forward(self, t, x):
        return self.gradient(concatenate_x_t(x, t))

    def gradient(self, s):
        # Equation 12
        out = self.resnet.jvp(s, self.w) + torch.nn.functional.linear(s, self.A.T @ self.A, self.b)
        return out[..., : -1]  # Remove the time prediction

    def hessian_trace(self, s: torch.Tensor, u0: torch.Tensor = None, z1: torch.Tensor = None):
        assert torch.all(torch.isfinite(s))
        assert torch.all(~torch.isnan(s))

        # Equation 14
        tr_first_term = self.resnet.hessian_trace(s, self.w, u0, z1)

        assert torch.all(torch.isfinite(tr_first_term))
        assert torch.all(~torch.isnan(tr_first_term))

        # E.T @ A ... remove last row (assuming E has d of d+1 standard basis vectors)
        # A @ E ... remove last column (assuming E has d of d+1 standard basis vectors)
        # tr_second_term = torch.trace((self.A.T @ self.A)[:-1, :-1])
        batch_size = s.shape[0]
        tr_second_term = torch.trace((self.A.T @ self.A)[:-1, :-1])
        tr_second_term = tr_second_term * torch.ones(batch_size, device=s.device)

        assert torch.all(torch.isfinite(tr_second_term))
        assert torch.all(~torch.isnan(tr_second_term))

        return tr_first_term + tr_second_term


class OTFlowODEFunction(ExactODEFunction):
    def __init__(self, n_dim, **kwargs):
        super().__init__(OTPotential(n_dim, **kwargs))

    def compute_log_det(self, t, x):
        return self.diffeq.hessian_trace(concatenate_x_t(x, t)).view(-1, 1)  # Need an empty dim at the end


class OTFlow(ExactContinuousBijection):
    """
    Optimal transport flow (OT-flow) architecture.

    Reference: Onken et al. "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport" (2021); https://arxiv.org/abs/2006.00104.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 ot_flow_kwargs: dict = None,
                 solver='rk4',
                 **kwargs):
        ot_flow_kwargs = ot_flow_kwargs or {}

        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = OTFlowODEFunction(n_dim, **ot_flow_kwargs)
        super().__init__(event_shape, diff_eq, solver=solver, **kwargs)
