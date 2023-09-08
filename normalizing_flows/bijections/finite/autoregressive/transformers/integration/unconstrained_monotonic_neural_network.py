import math

import torch
from typing import Union, Tuple, Callable, List

from normalizing_flows.bijections.finite.autoregressive.transformers.integration.base import Integration
from normalizing_flows.bijections.finite.autoregressive.util import gauss_legendre
from normalizing_flows.utils import sum_except_batch, pad_leading_dims


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
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_hidden_layers: int = 2,
                 hidden_dim: int = None):
        super().__init__(event_shape, g=self.neural_network_forward, c=torch.tensor(-100.0))
        self.n_hidden_layers = n_hidden_layers
        if hidden_dim is None:
            hidden_dim = max(5 * int(math.log(self.n_dim)), 4)
        self.hidden_dim = hidden_dim
        self.const = 1000  # for stability

        # weight, bias have self.hidden_dim elements
        self.n_input_params = 2 * self.hidden_dim

        # weight has self.hidden_dim elements, bias has just 1
        self.n_output_params = self.hidden_dim + 1

        # weight is a square matrix, bias is a vector
        self.n_hidden_params = (self.hidden_dim ** 2 + self.hidden_dim) * self.n_hidden_layers

        self._sampled_default_params = torch.randn(size=(self.n_parameters,)) / 1000

    @property
    def n_parameters(self) -> int:
        return self.n_input_params + self.n_output_params + self.n_hidden_params

    @property
    def default_parameters(self) -> torch.Tensor:
        return self._sampled_default_params

    def compute_parameters(self, h: torch.Tensor):
        batch_shape = h.shape[:-1]
        p0 = self.default_parameters

        # Input layer
        input_layer_defaults = pad_leading_dims(p0[:self.n_input_params], len(h.shape) - 1)
        input_layer_deltas = h[..., :self.n_input_params] / self.const
        input_layer_params = input_layer_defaults + input_layer_deltas
        input_layer_params = input_layer_params.view(*batch_shape, self.hidden_dim, 2)

        # Output layer
        output_layer_defaults = pad_leading_dims(p0[-self.n_output_params:], len(h.shape) - 1)
        output_layer_deltas = h[..., -self.n_output_params:] / self.const
        output_layer_params = output_layer_defaults + output_layer_deltas
        output_layer_params = output_layer_params.view(*batch_shape, 1, self.hidden_dim + 1)

        # Hidden layers
        hidden_layer_defaults = pad_leading_dims(
            p0[self.n_input_params:self.n_input_params + self.n_hidden_params],
            len(h.shape) - 1
        )
        hidden_layer_deltas = h[..., self.n_input_params:self.n_input_params + self.n_hidden_params] / self.const
        hidden_layer_params = hidden_layer_defaults + hidden_layer_deltas
        hidden_layer_params = torch.chunk(hidden_layer_params, chunks=self.n_hidden_layers, dim=-1)
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
    def reshape_tensors(x: torch.Tensor, h: List[torch.Tensor]):
        # batch_shape = get_batch_shape(x, self.event_shape)
        # batch_dims = int(torch.as_tensor(batch_shape).prod())
        # event_dims = int(torch.as_tensor(self.event_shape).prod())
        flattened_dim = int(torch.as_tensor(x.shape).prod())
        x_r = x.view(flattened_dim, 1, 1)
        h_r = [p.view(flattened_dim, *p.shape[-2:]) for p in h]
        return x_r, h_r

    def base_forward_1d(self, x: torch.Tensor, params: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_r, p_r = self.reshape_tensors(x, params)
        integral_flat = self.integral(x_r, p_r)
        log_det_flat = self.g(x_r, p_r).log()  # We can apply log since g is always positive
        output = integral_flat.view_as(x)
        log_det = log_det_flat.view_as(x)
        return output, log_det

    def inverse_1d(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.compute_parameters(h)
        z_r, p_r = self.reshape_tensors(z, params)
        x_flat = self.inverse_1d_without_log_det(z_r, p_r)
        outputs = x_flat.view_as(z)
        log_det = -self.g(x_flat, p_r).log().view_as(z)
        return outputs, log_det
