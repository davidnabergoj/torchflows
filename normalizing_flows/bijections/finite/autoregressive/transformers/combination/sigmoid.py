import math
from typing import Tuple, Union
import torch
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.base import Combination
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid_util import log_softmax, \
    log_sigmoid
from normalizing_flows.bijections.numerical_inversion import bisection_no_gradient
from normalizing_flows.utils import sum_except_batch


class Sigmoid(Transformer):
    """
    Applies z = inv_sigmoid(w.T @ sigmoid(a * x + b)) where a > 0, w > 0 and sum(w) = 1.
    Note: w, a, b are vectors, so multiplication a * x is broadcast.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]],
                 hidden_dim: int = None,
                 min_scale: float = 1e-3):
        super().__init__(event_shape)
        n_event_dims = int(torch.prod(torch.as_tensor(event_shape)))
        if hidden_dim is None:
            hidden_dim = max(4, 5 * int(math.log10(n_event_dims)))
        self.hidden_dim = hidden_dim

        self.min_scale = min_scale
        self.const = 1000
        self.default_u_a = math.log(math.e - 1 - self.min_scale)
        super().__init__(event_shape)

    @property
    def n_parameters(self) -> int:
        return 3 * self.hidden_dim

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=(self.n_parameters,))

    def extract_parameters(self, h: torch.Tensor):
        """
        h.shape = (*b, *e, self.n_parameters)
        """
        da = h[:, :self.hidden_dim]
        db = h[:, self.hidden_dim:self.hidden_dim * 2]
        dw = h[:, self.hidden_dim * 2:self.hidden_dim * 3]

        a = torch.nn.functional.softplus(self.default_u_a + da / self.const) + self.min_scale
        b = db / self.const
        w_pre = 0.0 + dw / self.const
        w = torch.softmax(w_pre, dim=-1)
        log_w = log_softmax(w_pre, dim=-1)
        return a, b, w, log_w

    def forward_1d(self, x, h):
        """
        x.shape = (n,)
        h.shape = (n, n_hidden * 3)

        Within the function:
        a.shape = (n, n_hidden)
        b.shape = (n, n_hidden)
        w.shape = (n, n_hidden)
        """
        a, b, w, log_w = self.extract_parameters(h)
        c = torch.sigmoid(a * x[:, None] + b)  # (*b, *e, n_hidden)
        d = torch.einsum('...i,...i->...', w, c)  # Softmax weighing -> (*b, *e)
        x = torch.log(d / (1 - d))  # Inverse sigmoid (*b, *e)

        log_t1 = (torch.log(d) - torch.log(1 - d))[:, None]  # (n, h)
        log_t2 = log_w  # (n, h)
        log_t3 = (log_sigmoid(c) + log_sigmoid(-c))  # (n, h)
        log_t4 = torch.log(a)  # (n, h)
        log_det = torch.sum(log_t1 + log_t2 + log_t3 + log_t4, dim=1)

        z = x
        return z, log_det

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        z_flat, log_det_flat = self.forward_1d(x_flat, h_flat)
        z = z_flat.view_as(x)
        log_det = sum_except_batch(log_det_flat.view_as(x), self.event_shape)
        return z, log_det

    def inverse_1d(self, z, h):
        def f(inputs):
            return self.forward_1d(inputs, h)

        x, log_det = bisection_no_gradient(f, z)
        return x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_flat = z.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        x_flat, log_det_flat = self.inverse_1d(z_flat, h_flat)
        x = x_flat.view_as(z)
        log_det = sum_except_batch(log_det_flat.view_as(z), self.event_shape)
        return x, log_det


class DenseSigmoid(Transformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class DeepSigmoid(Combination):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_hidden_layers: int = 2,
                 **kwargs):
        sigmoid_transforms = [Sigmoid(event_shape, **kwargs) for _ in range(n_hidden_layers)]
        super().__init__(event_shape, sigmoid_transforms)


class DeepDenseSigmoid(Combination):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_hidden_layers: int = 2,
                 **kwargs):
        sigmoid_transforms = [DenseSigmoid(event_shape, **kwargs) for _ in range(n_hidden_layers)]
        super().__init__(event_shape, sigmoid_transforms)
