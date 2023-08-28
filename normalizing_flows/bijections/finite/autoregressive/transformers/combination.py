import math
from typing import Tuple, Callable, Union, List
import torch
import torch.nn as nn
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.util import gauss_legendre, bisection
from normalizing_flows.utils import get_batch_shape, softmax_nd, sum_except_batch, log_softmax_nd, log_sigmoid


class Combination(Transformer):
    def __init__(self, event_shape: torch.Size, components: list[Transformer]):
        super().__init__(event_shape)
        self.components = components

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        # But this is not maintainable long-term.
        # We probably want ragged tensors with some parameter shape (akin to event and batch shapes).

        # Reshape h for easier access
        n_output_parameters = h.shape[-1] // len(self.components)
        h = torch.stack([
            h[..., i * n_output_parameters:(i + 1) * n_output_parameters]
            for i in range(len(self.components))
        ])

        assert len(h) == len(self.components)

        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(*batch_shape)
        for i in range(len(self.components)):
            x, log_det_increment = self.components[i].forward(x, h[i])
            log_det += log_det_increment

        return x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)

        # Reshape h for easier access
        n_output_parameters = h.shape[-1] // len(self.components)
        h = torch.stack([
            h[..., i * n_output_parameters:(i + 1) * n_output_parameters]
            for i in range(len(self.components))
        ])

        assert len(h) == len(self.components)

        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(*batch_shape)
        for i in range(len(self.components) - 1, -1, -1):
            z, log_det_increment = self.components[i].inverse(z, h[i])
            log_det += log_det_increment

        return z, log_det


class SigmoidTransform(Transformer):
    """
    Smallest invertible component of the deep sigmoidal networks.
    """

    def __init__(self, event_shape: torch.Size, hidden_dim: int = 8, epsilon: float = 1e-8):
        """

        :param event_shape: ...
        :param hidden_dim: hidden layer dimensionality. Authors recommend 8 or 16.
        """
        super().__init__(event_shape)
        self.hidden_dim = hidden_dim
        self.eps = epsilon

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward transformation is equal to y = log(w.T @ sigmoid(a * x + b)).

        :param x: inputs
        :param h: transformation parameters
        :return: outputs and log of the Jacobian determinant
        """
        # Reshape h for easier access to data
        base_shape = x.shape
        h = h.view(*base_shape, -1, 3)  # (a_unc, b, w_unc)

        a = torch.nn.functional.softplus(h[..., 0])  # Weights must be positive!
        b = h[..., 1]
        w_unc = h[..., 2]

        event_dims = tuple(range(len(x.shape)))[-len(self.event_shape):]
        extended_dims = tuple([*event_dims] + [len(x.shape)])
        w = softmax_nd(w_unc, dim=extended_dims)

        assert a.shape == b.shape == w.shape

        x_unsqueezed = x[..., None]  # Unsqueeze last dimension
        x_affine = a * x_unsqueezed + b
        x_sigmoid = torch.sigmoid(x_affine)
        x_convex = torch.sum(w * x_sigmoid, dim=-1)  # Sum over aux dim (dot product)
        x_convex_clipped = x_convex * (1 - self.eps) + self.eps * 0.5
        y = torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)

        log_det = log_softmax_nd(w_unc, extended_dims) + log_sigmoid(x_affine) + log_sigmoid(-x_affine) + torch.log(a)
        log_det = torch.logsumexp(log_det, -1)  # LSE over aux dim
        log_det += math.log(1 - self.eps) - torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)
        log_det = sum_except_batch(log_det, self.event_shape)
        # log_det = sum_except_batch(torch.log(torch.sum((1 - x_sigmoid) * a, dim=-1)), self.event_shape)
        return y, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The original paper introducing deep sigmoidal networks did not provide an analytic inverse.
        # Inverting the transformation can be done numerically.
        raise NotImplementedError


class InverseSigmoidTransform(SigmoidTransform):
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(z, h)


class DenseSigmoidTransform(Transformer):
    def __init__(self,
                 event_shape: torch.Size,
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 8,
                 epsilon: float = 1e-8):
        """

        :param event_shape: ...
        :param hidden_dim: hidden layer dimensionality. Authors recommend 8 or 16.
        """
        super().__init__(event_shape)
        self.hidden_dim = hidden_dim
        self.eps = epsilon

        self.u_ = nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.nn.functional.softplus(h[..., 0])  # Weights must be positive!
        b = h[..., 1]
        w_unc = h[..., 2]
        u_unc = h[..., 3]

        event_dims = tuple(range(len(x.shape)))[-len(self.event_shape):]
        extended_dims = tuple([*event_dims] + [len(x.shape)])
        w = softmax_nd(w_unc, dim=extended_dims)
        u = softmax_nd(u_unc, dim=extended_dims)

        assert a.shape == b.shape == w.shape == u.shape

        x_unsqueezed = x[..., None]  # Unsqueeze last dimension
        x_affine = a * x_unsqueezed + b
        x_sigmoid = torch.sigmoid(x_affine)
        x_convex = torch.sum(w * x_sigmoid, dim=-1)  # Sum over aux dim (dot product)
        x_convex_clipped = x_convex * (1 - self.eps) + self.eps * 0.5
        y = torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)

        log_det = log_softmax_nd(w_unc, extended_dims) + log_sigmoid(x_affine) + log_sigmoid(-x_affine) + torch.log(a)
        log_det = torch.logsumexp(log_det, -1)  # LSE over aux dim
        log_det += math.log(1 - self.eps) - torch.log(x_convex_clipped) - torch.log(1 - x_convex_clipped)
        log_det = sum_except_batch(log_det, self.event_shape)
        # log_det = sum_except_batch(torch.log(torch.sum((1 - x_sigmoid) * a, dim=-1)), self.event_shape)
        return y, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The original paper introducing deep sigmoidal networks did not provide an analytic inverse.
        # Inverting the transformation can be done numerically.
        raise NotImplementedError


class DeepSigmoidNetwork(Combination):
    """
    Deep sigmoidal network transformer as proposed in "Huang et al. Neural Autoregressive Flows (2018)".
    """

    def __init__(self, event_shape: torch.Size, n_layers: int = 2, **kwargs):
        super().__init__(event_shape, [SigmoidTransform(event_shape, **kwargs) for _ in range(n_layers)])


class InverseDeepSigmoidNetwork(Combination):
    def __init__(self, event_shape: torch.Size, n_layers: int = 2, **kwargs):
        super().__init__(event_shape, [InverseSigmoidTransform(event_shape, **kwargs) for _ in range(n_layers)])


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
