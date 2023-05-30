from typing import Tuple, List

import torch

import torch.nn as nn


class Bijection(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Bijection class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward bijection map.
        Returns the output vector and the log jacobian determinant of the forward transform.

        :param x:
        :return:
        """
        raise NotImplementedError

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse bijection map.
        Returns the output vector and the log jacobian determinant of the inverse transform.

        :param z:
        :return:
        """
        raise NotImplementedError


class BijectiveComposition(Bijection):
    def __init__(self, layers: List[Bijection]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det_layer = layer(x)
            log_det += log_det_layer
        z = x
        return z, log_det

    def inverse(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers[::-1]:
            z, log_det_layer = layer.inverse(z)
            log_det += log_det_layer
        x = z
        return x, log_det
