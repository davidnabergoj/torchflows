from typing import Union, Tuple

import torch

from torchflows.bijections.base import BijectiveComposition
from torchflows.bijections.finite.autoregressive.transformers.linear.affine import Affine
from torchflows.bijections.finite.residual.base import ResidualComposition
from torchflows.bijections.finite.residual.iterative import InvertibleResNetBlock, ResFlowBlock
from torchflows.bijections.finite.residual.proximal import ProximalResFlowBlock
from torchflows.bijections.finite.residual.planar import Planar
from torchflows.bijections.finite.residual.radial import Radial
from torchflows.bijections.finite.residual.sylvester import Sylvester


class InvertibleResNet(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        blocks = [
            InvertibleResNetBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)


class ResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        blocks = [
            ResFlowBlock(event_shape=event_shape, context_shape=context_shape, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)


class ProximalResFlow(ResidualComposition):
    def __init__(self, event_shape, context_shape=None, n_layers: int = 16, **kwargs):
        blocks = [
            ProximalResFlowBlock(event_shape=event_shape, context_shape=context_shape, gamma=0.01, **kwargs)
            for _ in range(n_layers)
        ]
        super().__init__(blocks)


class PlanarFlow(BijectiveComposition):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_layers: int = 2):
        if n_layers < 1:
            raise ValueError(f"Flow needs at least one layer, but got {n_layers}")
        super().__init__(event_shape, [
            Affine(event_shape),
            *[Planar(event_shape) for _ in range(n_layers)],
            Affine(event_shape)
        ])


class RadialFlow(BijectiveComposition):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_layers: int = 2):
        if n_layers < 1:
            raise ValueError(f"Flow needs at least one layer, but got {n_layers}")
        super().__init__(event_shape, [
            Affine(event_shape),
            *[Radial(event_shape) for _ in range(n_layers)],
            Affine(event_shape)
        ])


class SylvesterFlow(BijectiveComposition):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_layers: int = 2, **kwargs):
        if n_layers < 1:
            raise ValueError(f"Flow needs at least one layer, but got {n_layers}")
        super().__init__(event_shape, [
            Affine(event_shape),
            *[Sylvester(event_shape, **kwargs) for _ in range(n_layers)],
            Affine(event_shape)
        ])
