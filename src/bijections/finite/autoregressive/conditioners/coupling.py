from typing import Dict

import torch
import torch.nn as nn

from src.bijections.finite.autoregressive.conditioners.base import Conditioner


class Coupling(Conditioner):
    def __init__(self,
                 transform: nn.Module,
                 constants: torch.Tensor,
                 constant_dims: torch.Tensor,
                 n_dim: int):
        """
        Coupling conditioner.


        Note: Always treats the first n_dim // 2 dimensions as constant. Shuffling is handled in Permutation bijections.

        :param transform: module which predicts transformer parameters.
            Input dimension should be len(constant_dims).
            Output dimension should be n_dim - len(constant_dims).
        :param constants:
        :param constant_dims:
        """
        super().__init__()
        self.transform = transform
        self.constant_dims = constant_dims
        self.transformed_dims = torch.arange(n_dim)[~torch.isin(torch.arange(n_dim), constant_dims)]
        self.n_dim = n_dim

        self.constants = torch.stack([constants for _ in range(len(constant_dims))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict transformer parameters for output dimensions
        tmp = self.transform(x[:, self.constant_dims])
        n_data = tmp.shape[0]
        n_parameters = tmp.shape[-1]

        # Create full parameter tensor
        h = torch.empty(size=(n_data, self.n_dim, n_parameters), dtype=x.dtype)

        # Fill the parameter tensor with predicted values
        h[:, self.transformed_dims] = tmp
        h[:, self.constant_dims] = torch.stack([self.constants for _ in range(n_data)])

        return h
