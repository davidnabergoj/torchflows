import math
from typing import Tuple, Callable, Union, List
import torch
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.base import Combination
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid_util import log_softmax, \
    log_sigmoid
from normalizing_flows.bijections.finite.autoregressive.util import gauss_legendre, bisection
from normalizing_flows.utils import get_batch_shape, sum_except_batch


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


class DenseSigmoid(Transformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
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
        sigmoid_transforms = [Sigmoid(event_shape, **kwargs) for _ in range(n_hidden_layers)]
        super().__init__(event_shape, sigmoid_transforms)


class MonotonicTransformer(Transformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], bound: float = 100.0, eps: float = 1e-6):
        super().__init__(event_shape)
        self.bound = bound
        self.eps = eps

    def integral(self, x, h) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, h: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            x = x.requires_grad_()
            z = self.integral(x, h)
        jacobian = torch.autograd.grad(z, x, torch.ones_like(z), create_graph=True)[0]
        log_det = jacobian.log()
        return z, log_det

    def inverse(self, z: torch.Tensor, h: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = bisection(
            f=self.integral,
            y=z,
            a=torch.full_like(z, -self.bound),
            b=torch.full_like(z, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            h=h
        )
        return x, -self.forward(x, h)[1]


class UnconstrainedMonotonicTransformer(MonotonicTransformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], g: Callable, c: torch.Tensor, n_quad: int = 32,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.g = g
        self.c = c
        self.n = n_quad

    def integral(self, x, h) -> torch.Tensor:
        return gauss_legendre(f=self.g, a=torch.zeros_like(x), b=x, n=self.n, h=h) + self.c

    def forward(self, x: torch.Tensor, h: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.integral(x, h), self.g(x, h).log()


class UnconstrainedMonotonicNeuralNetwork(UnconstrainedMonotonicTransformer):

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_hidden_layers: int, hidden_dim: int):
        # TODO make it so that predicted parameters only affect the final linear layer instead of the entire neural
        #  network. That is much more easily optimized. The rest of the NN are trainable parameters.
        super().__init__(event_shape, g=self.neural_network_forward, c=torch.tensor(-100.0))
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

    def reshape_conditioner_outputs(self, h):
        batch_shape = h.shape[:-1]
        input_layer_params = h[..., :2 * self.hidden_dim].view(*batch_shape, self.hidden_dim, 2)
        output_layer_params = h[..., -(self.hidden_dim + 1):].view(*batch_shape, 1, self.hidden_dim + 1)
        hidden_layer_params = torch.chunk(
            h[..., 2 * self.hidden_dim:(h.shape[-1] - (self.hidden_dim + 1))],
            chunks=self.n_hidden_layers,
            dim=-1
        )
        hidden_layer_params = [
            layer.view(*batch_shape, self.hidden_dim, self.hidden_dim + 1)
            for layer in hidden_layer_params
        ]
        return [input_layer_params, *hidden_layer_params, output_layer_params]

    @staticmethod
    def neural_network_forward(inputs, parameters: List[torch.Tensor]):
        # inputs.shape = (batch_size, 1, 1)
        # Each element of parameters is a (batch_size, n_out, n_in + 1) tensor comprised of matrices for the linear
        # projection and bias vectors.
        assert len(inputs.shape) == 3
        assert inputs.shape[1:] == (1, 1)
        batch_size = inputs.shape[0]
        assert all(p.shape[0] == batch_size for p in parameters)
        assert all(len(p.shape) == 3 for p in parameters)
        assert parameters[0].shape[2] == 1 + 1  # input dimension should be 1 (we also acct for the bias)
        assert parameters[-1].shape[1] == 1  # output dimension should be 1

        # Neural network pass
        out = inputs
        for param in parameters[:-1]:
            weight_matrices = param[:, :, :-1]
            bias_vectors = param[:, :, -1][..., None]
            out = torch.bmm(weight_matrices, out) + bias_vectors
            out = torch.tanh(out)

        # Final output
        weight_matrices = parameters[-1][:, :, :-1]
        bias_vectors = parameters[-1][:, :, -1][..., None]
        out = torch.bmm(weight_matrices, out) + bias_vectors
        out = 1 + torch.nn.functional.elu(out)
        return out

    @staticmethod
    def flatten_tensors(x: torch.Tensor, h: List[torch.Tensor]):
        # batch_shape = get_batch_shape(x, self.event_shape)
        # batch_dims = int(torch.as_tensor(batch_shape).prod())
        # event_dims = int(torch.as_tensor(self.event_shape).prod())
        flattened_dim = int(torch.as_tensor(x.shape).prod())
        x_flat = x.view(flattened_dim, 1, 1)
        h_flat = [h[i].view(flattened_dim, *h[i].shape[-2:]) for i in range(len(h))]
        return x_flat, h_flat

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.reshape_conditioner_outputs(h)
        x_reshaped, h_reshaped = self.flatten_tensors(x, h)

        integral_flat = self.integral(x_reshaped, h_reshaped)
        log_det_flat = self.g(x_reshaped, h_reshaped).log()  # We can apply log since g is always positive

        output = integral_flat.view_as(x)
        log_det = sum_except_batch(log_det_flat.view_as(x), self.event_shape)

        return output, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.reshape_conditioner_outputs(h)
        z_flat, h_flat = self.flatten_tensors(z, h)

        x_flat = bisection(
            f=self.integral,
            y=z_flat,
            a=torch.full_like(z_flat, -self.bound),
            b=torch.full_like(z_flat, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            h=h_flat
        )

        outputs = x_flat.view_as(z)
        log_det = sum_except_batch(-self.g(x_flat, h_flat).log().view_as(z), self.event_shape)

        return outputs, log_det
