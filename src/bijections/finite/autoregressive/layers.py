from typing import Tuple

import torch
import torch.nn as nn

from src.bijections.finite.autoregressive.conditioners.coupling import Coupling
from src.bijections.finite.autoregressive.transformers.affine import Affine
from src.bijections.finite.base import Bijection


class AffineCoupling(Bijection):
    def __init__(self, n_dim, constant_dims, conditioner_transform: nn.Module, scale_transform: callable = torch.exp):
        super().__init__()
        default_log_scale = 0.0
        default_shift = 0.0
        self.scale_transform = scale_transform
        self.transformer = Affine(scale_transform=scale_transform)
        self.conditioner = Coupling(
            transform=conditioner_transform,
            constants=torch.tensor([default_log_scale, default_shift]),
            constant_dims=constant_dims,
            n_dim=n_dim
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(x)
        z, log_det = self.transformer(x, h)
        return z, log_det

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(z)
        x, log_det = self.transformer.inverse(z, h)
        return x, log_det


class LinearAffineCoupling(AffineCoupling):
    def __init__(self, n_dim: int, **kwargs):
        assert n_dim >= 2

        n_transformer_parameters = 2

        # Set up the input and output dimensions
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the linear conditioner
        lin_cond = nn.Sequential(
            nn.Linear(n_constant_dims, n_transformed_dims * n_transformer_parameters),
            nn.Unflatten(dim=1, unflattened_size=(n_transformed_dims, n_transformer_parameters))
        )

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=lin_cond,
            **kwargs
        )


class FeedForwardAffineCoupling(AffineCoupling):
    def __init__(self, n_dim: int, n_hidden: int = 100, n_layers: int = 4, **kwargs):
        assert n_dim >= 2

        n_transformer_parameters = 2

        # Set up the input and output dimensions
        constant_dims = torch.arange(n_dim // 2)
        n_constant_dims = len(constant_dims)
        n_transformed_dims = n_dim - n_constant_dims

        # Create the conditioner neural network
        layers = [nn.Linear(n_constant_dims, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        layers.append(nn.Linear(n_hidden, n_transformed_dims * n_transformer_parameters))
        layers.append(nn.Unflatten(dim=1, unflattened_size=(n_transformed_dims, n_transformer_parameters)))
        network = nn.Sequential(*layers)

        super().__init__(
            n_dim=n_dim,
            constant_dims=constant_dims,
            conditioner_transform=network,
            **kwargs
        )
