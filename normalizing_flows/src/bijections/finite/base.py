from typing import Tuple, List, Union

import torch

import torch.nn as nn

from normalizing_flows.src.utils import get_batch_shape


class Bijection(nn.Module):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        """
        Bijection class.
        """
        super().__init__()
        self.event_shape = event_shape

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward bijection map.
        Returns the output vector and the log Jacobian determinant of the forward transform.

        :param x: input array with shape (*batch_shape, *event_shape).
        :param context: context array with shape (*batch_shape, context_dim).
        :return: output array and log determinant. The output array has shape (*batch_shape, *event_shape); the log
            determinant has shape (*batch_shape,).
        """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse bijection map.
        Returns the output vector and the log Jacobian determinant of the inverse transform.

        :param z: input array with shape (*batch_shape, *event_shape).
        :param context: context array with shape (*batch_shape, context_dim).
        :return: output array and log determinant. The output array has shape (*batch_shape, *event_shape); the log
            determinant has shape (*batch_shape,).
        """
        raise NotImplementedError


class BijectiveComposition(Bijection):
    def __init__(self, event_shape: torch.Size, layers: List[Bijection]):
        super().__init__(event_shape=event_shape)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(*get_batch_shape(x, event_shape=self.event_shape), device=x.device)
        for layer in self.layers:
            x, log_det_layer = layer(x, context=context)
            log_det += log_det_layer
        z = x
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(*get_batch_shape(z, event_shape=self.event_shape), device=z.device)
        for layer in self.layers[::-1]:
            z, log_det_layer = layer.inverse(z, context=context)
            log_det += log_det_layer
        x = z
        return x, log_det
