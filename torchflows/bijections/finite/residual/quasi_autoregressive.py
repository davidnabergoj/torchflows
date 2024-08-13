from typing import Union, Tuple, List

import torch
import torch.nn as nn

from torchflows.bijections.finite.residual.base import ResidualBijection


class AffineQuasiMADELayerSet(nn.Module):
    def __init__(self, k: int, n_outputs: int):
        super().__init__()
        assert k >= 1
        assert n_outputs >= 1

        n_inputs = k
        # n_outputs = k in the next Quasi MADE layer

        divisor = n_inputs * n_outputs
        self.weight = nn.Parameter(torch.randn(size=(n_outputs, n_inputs)) / divisor)
        self.bias = nn.Parameter(torch.randn(size=(n_outputs,)) / divisor)

    def forward(self, x):
        # compute y and jacobian of the layer transformation
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        jac = self.weight
        return y, jac


class AffineQuasiMADELayer(nn.Module):
    def __init__(self, n_sets: int, n_set_inputs: int, n_set_outputs: int):
        super().__init__()
        self.n_sets = n_sets
        self.n_set_inputs = n_set_inputs  # this k
        self.n_set_outputs = n_set_outputs  # next k
        self.set_transforms = nn.ModuleList([
            AffineQuasiMADELayerSet(self.n_set_inputs, self.n_set_outputs)
            for _ in range(n_sets)
        ])

    def forward(self, x):
        # x.shape = (batch_size, k * n_sets)
        # Partition into sets
        x_chunks = torch.chunk(x, chunks=self.n_sets, dim=-1)
        out = []
        for i in range(self.n_sets):
            transform = self.set_transforms[i]
            y_i, jac_i = transform(x_chunks[i])
            out.append((y_i, jac_i))
        return out


class AffineQuasiMADE(nn.Module):
    def __init__(self, n_layers: int, k: List[int]):
        super().__init__()
        assert len(k) == n_layers
        assert k[0] == k[-1] == 1
        assert n_layers >= 2
        layers = []
        for i in range(n_layers - 1):
            layers.append(AffineQuasiMADELayer(k[i], k[i + 1]))
        self.layers = nn.ParameterList(layers)

    def forward(self, x):
        # TODO put the jacobian directly into autograd
        jac = None
        for layer in self.layers:
            layer_ret = layer(x)
            x = [layer_ret[i][0] for i in range(len(layer_ret))]
            jac_update = [layer_ret[i][1] for i in range(len(layer_ret))]

            x = torch.cat(x, dim=-1)

            if jac is None:
                jac = jac_update
            else:
                jac = [jac[i] @ jac_update[i] for i in range(len(jac_update))]
        return x, jac


class QuasiAutoregressiveFlowBlock(ResidualBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], sigma: float = 0.7):
        super().__init__(event_shape)
        self.log_theta = nn.Parameter(torch.randn())
        self.sigma = sigma
        self.g = ...
