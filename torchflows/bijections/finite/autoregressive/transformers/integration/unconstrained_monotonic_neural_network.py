import math

import torch
from typing import Union, Tuple, Callable, List

from torchflows.bijections.finite.autoregressive.transformers.integration.base import Integration
from torchflows.bijections.finite.autoregressive.util import gauss_legendre
from torchflows.utils import sum_except_batch, pad_leading_dims


class UnconstrainedMonotonicTransformer(Integration):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], g: Callable, c: torch.Tensor, n_quad: int = 32,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.g = g  # g takes as input a list of torch tensors
        self.c = c
        self.n = n_quad

    def integral(self, x: torch.Tensor, h: List[torch.Tensor]) -> torch.Tensor:
        return gauss_legendre(f=self.g, a=torch.zeros_like(x), b=x, n=self.n, h=h) + self.c

    def base_forward_1d(self, x: torch.Tensor, params: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.integral(x, params), self.g(x, params).log()


class UnconstrainedMonotonicNeuralNetwork(UnconstrainedMonotonicTransformer):
    """
    Unconstrained monotonic neural network transformer.

    The unconstrained monotonic neural network is a neural network with positive weights and positive activation
     function derivatives. These two conditions ensure its invertibility.
    """
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_hidden_layers: int = None,
                 hidden_dim: int = None):
        super().__init__(event_shape, g=self.neural_network_forward, c=torch.tensor(-100.0))

        if n_hidden_layers is None:
            n_hidden_layers = 1
        self.n_hidden_layers = n_hidden_layers

        if hidden_dim is None:
            hidden_dim = max(int(math.log(self.n_dim)), 4)
        self.hidden_dim = hidden_dim

        self.const = 1  # for stability

        # weight, bias have self.hidden_dim elements
        self.n_input_params = 2 * self.hidden_dim

        # weight has self.hidden_dim elements, bias has just 1
        self.n_output_params = self.hidden_dim + 1

        # weight is a square matrix, bias is a vector
        self.n_hidden_params = (self.hidden_dim ** 2 + self.hidden_dim) * self.n_hidden_layers

        self._sampled_default_params = torch.randn(size=(self.n_dim, *self.parameter_shape_per_element)) / 1000

    @property
    def parameter_shape_per_element(self) -> Union[torch.Size, Tuple]:
        return (self.n_input_params + self.n_output_params + self.n_hidden_params,)

    @property
    def default_parameters(self) -> torch.Tensor:
        return self._sampled_default_params

    def compute_parameters(self, h: torch.Tensor):
        p0 = self.default_parameters
        batch_size = h.shape[0]
        n_events = batch_size // p0.shape[0]

        # Input layer
        input_layer_defaults = p0[..., :self.n_input_params].repeat(n_events, 1)
        input_layer_deltas = h[..., :self.n_input_params] / self.const
        input_layer_params = input_layer_defaults + input_layer_deltas
        input_layer_params = input_layer_params.view(batch_size, self.hidden_dim, 2)

        # Output layer
        output_layer_defaults = p0[..., -self.n_output_params:].repeat(n_events, 1)
        output_layer_deltas = h[..., -self.n_output_params:] / self.const
        output_layer_params = output_layer_defaults + output_layer_deltas
        output_layer_params = output_layer_params.view(batch_size, 1, self.hidden_dim + 1)

        # Hidden layers
        hidden_layer_defaults = p0[..., self.n_input_params:self.n_input_params + self.n_hidden_params].repeat(n_events, 1)
        hidden_layer_deltas = h[..., self.n_input_params:self.n_input_params + self.n_hidden_params] / self.const
        hidden_layer_params = hidden_layer_defaults + hidden_layer_deltas
        hidden_layer_params = torch.chunk(hidden_layer_params, chunks=self.n_hidden_layers, dim=-1)
        hidden_layer_params = [
            layer.view(batch_size, self.hidden_dim, self.hidden_dim + 1)
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

    def base_forward_1d(self, x: torch.Tensor, params: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_r = x.view(-1, 1, 1)
        output = self.integral(x_r, params).view_as(x)
        log_det = self.g(x_r, params).log().view_as(x)  # We can apply log since g is always positive
        return output, log_det

    def inverse_1d(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.compute_parameters(h)
        z_r = z.view(-1, 1, 1)
        x_flat = self.inverse_1d_without_log_det(z_r, params)
        outputs = x_flat.view_as(z)
        log_det = -self.g(x_flat, params).log().view_as(z)
        return outputs, log_det
