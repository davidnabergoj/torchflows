import math
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.base import Combination
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid_util import log_softmax, \
    log_sigmoid, log_dot
from normalizing_flows.bijections.numerical_inversion import bisection_no_gradient
from normalizing_flows.utils import sum_except_batch, get_batch_shape


# As defined in the NAF paper

def inverse_sigmoid(p):
    return torch.log(p / (1 - p))


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
        h.shape = (batch_size, self.n_parameters)
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
        x.shape = (batch_size,)
        h.shape = (batch_size, hidden_size * 3)

        Within the function:
        a.shape = (batch_size, hidden_size)
        b.shape = (batch_size, hidden_size)
        w.shape = (batch_size, hidden_size)
        """
        a, b, w, log_w = self.extract_parameters(h)
        c = torch.sigmoid(a * x[:, None] + b)  # (batch_size, n_hidden)
        d = torch.einsum('...i,...i->...', w, c)  # Softmax weighing -> (batch_size,)
        x = inverse_sigmoid(d)  # Inverse sigmoid ... (batch_size,)

        log_t1 = (torch.log(d) - torch.log(1 - d))[:, None]  # (batch_size, hidden_size)
        log_t2 = log_w  # (batch_size, hidden_size)
        log_t3 = (log_sigmoid(c) + log_sigmoid(-c))  # (batch_size, hidden_size)
        log_t4 = torch.log(a)  # (batch_size, hidden_size)
        log_det = torch.sum(log_t1 + log_t2 + log_t3 + log_t4, dim=1)

        z = x
        return z, log_det

    def inverse_1d(self, z, h):
        def f(inputs):
            return self.forward_1d(inputs, h)

        x, log_det = bisection_no_gradient(f, z)
        return x, log_det

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        z_flat, log_det_flat = self.forward_1d(x_flat, h_flat)
        z = z_flat.view_as(x)
        log_det = sum_except_batch(log_det_flat.view_as(x), self.event_shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_flat = z.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        x_flat, log_det_flat = self.inverse_1d(z_flat, h_flat)
        x = x_flat.view_as(z)
        log_det = sum_except_batch(log_det_flat.view_as(z), self.event_shape)
        return x, log_det


class DenseSigmoidInnerTransform(nn.Module):
    def __init__(self, input_size, output_size, min_scale: float = 1e-3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.min_scale = min_scale
        self.const = 1000
        self.default_u_a = math.log(math.e - 1 - self.min_scale)

    @property
    def n_parameters(self):
        return self.output_size ** 2 + 2 * self.output_size + self.output_size * self.input_size

    def extract_parameters(self, h: torch.Tensor):
        """
        h.shape = (batch_size, self.n_parameters)
        """
        assert len(h.shape) == 2
        batch_size = len(h)

        da = h[:, :self.output_size]
        db = h[:, self.output_size:self.output_size * 2]
        dw = h[:, self.output_size * 2:self.output_size * 2 + self.output_size ** 2]
        du = h[:, self.output_size * 2 + self.output_size ** 2:]

        du = du.view(batch_size, self.output_size, self.input_size)
        dw = dw.view(batch_size, self.output_size, self.output_size)

        u_pre = 0.0 + du / self.const
        u = torch.softmax(u_pre, dim=-1)
        log_u = log_softmax(u_pre, dim=-1)

        a = torch.nn.functional.softplus(self.default_u_a + da / self.const) + self.min_scale
        b = db / self.const
        w_pre = 0.0 + dw / self.const
        w = torch.softmax(w_pre, dim=-1)
        log_w = log_softmax(w_pre, dim=-1)
        return a, b, w, log_w, u, log_u

    def forward_1d(self, x, h):
        # Compute y = inv_sig(w @ sig(a * u @ x + b))
        # h.shape = (batch_size, n_parameters)

        # Within the function:
        # x.shape = (batch_size, input_size)
        # a.shape = (batch_size, output_size)
        # b.shape = (batch_size, output_size)
        # w.shape = (batch_size, output_size, output_size)
        # log_w.shape = (batch_size, output_size, output_size)
        # u.shape = (batch_size, output_size, input_size)
        # log_u.shape = (batch_size, output_size, input_size)

        # Return
        # y.shape = (batch_size, output_size)
        # log_det.shape = (batch_size,)

        a, b, w, log_w, u, log_u = self.extract_parameters(h)

        ux = torch.einsum('boi,bi->bo', u, x)  # (batch_size, output_size)
        c = torch.sigmoid(a * ux + b)  # (batch_size, output_size)
        d = torch.einsum('bij,bj->bi', w, c)  # Softmax weighing -> (batch_size, output_size)
        x = inverse_sigmoid(d)  # Inverse sigmoid (batch_size, output_size)

        log_t1 = (torch.log(d) - torch.log(1 - d))[:, :, None]  # (batch_size, output_size, 1)
        log_t2 = log_w  # (batch_size, output_size, output_size)
        log_t3 = (log_sigmoid(c) + log_sigmoid(-c))[:, None, :]  # (batch_size, 1, output_size)
        log_t4 = torch.log(a)[:, None, :]  # (batch_size, 1, output_size)

        m1 = (log_t1 + log_t2 + log_t3 + log_t4)[:, :, :, None]  # (batch_size, output_size, output_size, 1)
        m2 = log_u[:, None, :, :]  # (batch_size, 1, output_size, input_size)
        log_det = torch.sum(log_dot(m1, m2), dim=(1, 2, 3))  # (batch_size,)

        z = x
        return z, log_det


class DenseSigmoid(Transformer):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_dense_layers: int = 1,
                 hidden_size: int = 30):
        super().__init__(event_shape)
        self.n_dense_layers = n_dense_layers
        layers = [
            DenseSigmoidInnerTransform(self.n_dim, hidden_size),
            *[DenseSigmoidInnerTransform(hidden_size, hidden_size) for _ in range(n_dense_layers)],
            DenseSigmoidInnerTransform(hidden_size, self.n_dim),
        ]
        self.layers = nn.ModuleList(layers)

    @property
    def n_parameters(self) -> int:
        return sum([layer.n_parameters for layer in self.layers])

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=(self.n_parameters,))  # TODO set up parametrization with deltas so this holds

    def split_parameters(self, h):
        # split parameters h into parameters for several layers
        # h.shape == (*batch_shape, *event_shape, n_parameters)
        split_parameters = torch.split(h, split_size_or_sections=[layer.n_parameters for layer in self.layers], dim=-1)
        return split_parameters

    def forward_1d(self, x_flat, h_split_flat: List[torch.Tensor]):
        log_det_flat = None
        for i in range(len(self.layers)):
            x_flat, log_det_flat_inc = self.layers[i].forward_1d(x_flat, h_split_flat[i])
            if log_det_flat is None:
                log_det_flat = log_det_flat_inc
            else:
                log_det_flat += log_det_flat_inc
        z_flat = x_flat
        return z_flat, log_det_flat

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape == (*batch_shape, *event_shape)
        # h.shape == (*batch_shape, n_parameters)
        h_split = self.split_parameters(h)
        event_size = self.n_dim
        batch_size = int(torch.prod(torch.as_tensor(get_batch_shape(x, self.event_shape))))
        x_flat = x.view(batch_size, event_size)
        h_split_flat = [h_split[i].view(batch_size, -1) for i in range(len(h_split))]

        z_flat, log_det_flat = self.forward_1d(x_flat, h_split_flat)

        log_det = log_det_flat.view(batch_size)
        z = z_flat.view_as(x)
        return z, log_det

    def inverse_1d(self, z, h):
        def f(inputs):
            return self.forward_1d(inputs, h)

        x, log_det = bisection_no_gradient(f, z)
        return x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape == (*batch_shape, *event_shape)
        # h.shape == (*batch_shape, n_parameters)
        h_split = self.split_parameters(h)
        event_size = self.n_dim
        batch_size = int(torch.prod(torch.as_tensor(get_batch_shape(z, self.event_shape))))
        z_flat = z.view(batch_size, event_size)
        h_split_flat = [h_split[i].view(batch_size, -1) for i in range(len(h_split))]

        x_flat, log_det_flat = self.inverse_1d(z_flat, h_split_flat)

        log_det = log_det_flat.view(batch_size)
        z = x_flat.view_as(z)
        return z, log_det


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
