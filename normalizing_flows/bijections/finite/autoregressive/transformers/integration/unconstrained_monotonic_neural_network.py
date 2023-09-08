import math

import torch
from typing import Union, Tuple, Callable, List

from normalizing_flows.bijections.finite.autoregressive.transformers.integration.base import Integration
from normalizing_flows.bijections.finite.autoregressive.util import gauss_legendre, bisection
from normalizing_flows.utils import sum_except_batch


class UnconstrainedMonotonicTransformer(Integration):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], g: Callable, c: torch.Tensor, n_quad: int = 32,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.g = g  # g takes as input a list of torch tensors
        self.c = c
        self.n = n_quad

    def integral(self, x: torch.Tensor, h: List[torch.Tensor]) -> torch.Tensor:
        return gauss_legendre(f=self.g, a=torch.zeros_like(x), b=x, n=self.n, h=h) + self.c

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.split_h(h)
        return self.integral(x, params), self.g(x, params).log()


class UnconstrainedMonotonicNeuralNetwork(UnconstrainedMonotonicTransformer):

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_hidden_layers: int, hidden_dim: int):
        # TODO make it so that predicted parameters only affect the final linear layer instead of the entire neural
        #  network. That is much more easily optimized. The rest of the NN are globally trainable parameters.
        super().__init__(event_shape, g=self.neural_network_forward, c=torch.tensor(-100.0))
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

    def split_h(self, h: torch.Tensor):
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
        h = self.split_h(h)
        x_reshaped, h_reshaped = self.flatten_tensors(x, h)

        integral_flat = self.integral(x_reshaped, h_reshaped)
        log_det_flat = self.g(x_reshaped, h_reshaped).log()  # We can apply log since g is always positive

        output = integral_flat.view_as(x)
        log_det = sum_except_batch(log_det_flat.view_as(x), self.event_shape)

        return output, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.split_h(h)
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
