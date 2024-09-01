from typing import Union, Tuple

import torch

from torchflows.bijections.finite.residual.base import ResidualArchitecture
from torchflows.bijections.finite.residual.iterative import InvertibleResNetBlock, ResFlowBlock
from torchflows.bijections.finite.residual.proximal import ProximalResFlowBlock
from torchflows.bijections.finite.residual.planar import Planar
from torchflows.bijections.finite.residual.radial import Radial
from torchflows.bijections.finite.residual.sylvester import Sylvester


class InvertibleResNet(ResidualArchitecture):
    """Invertible residual network (i-ResNet) architecture.

    Reference: Behrmann et al. "Invertible Residual Networks" (2019); https://arxiv.org/abs/1811.00995.
    """

    def __init__(self, event_shape, **kwargs):
        super().__init__(event_shape, InvertibleResNetBlock, **kwargs)


class ResFlow(ResidualArchitecture):
    """Residual flow (ResFlow) architecture.

    Reference: Chen et al. "Residual Flows for Invertible Generative Modeling" (2020); https://arxiv.org/abs/1906.02735.
    """

    def __init__(self, event_shape, **kwargs):
        super().__init__(event_shape, ResFlowBlock, **kwargs)


class ProximalResFlow(ResidualArchitecture):
    """Proximal residual flow architecture.

    Reference: Hertrich "Proximal Residual Flows for Bayesian Inverse Problems" (2022); https://arxiv.org/abs/2211.17158.
    """

    def __init__(self, event_shape, **kwargs):
        if 'layer_kwargs' not in kwargs:
            kwargs['layer_kwargs'] = {}
        if 'gamma' not in kwargs['layer_kwargs']:
            kwargs['layer_kwargs']['gamma'] = 0.01
        super().__init__(event_shape, ProximalResFlowBlock, **kwargs)


class PlanarFlow(ResidualArchitecture):
    """Planar flow architecture.

    Note: this model currently supports only one-way transformations.

    Reference: Rezende and Mohamed "Variational Inference with Normalizing Flows" (2016); https://arxiv.org/abs/1505.05770.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 **kwargs):
        super().__init__(event_shape, Planar, **kwargs)


class RadialFlow(ResidualArchitecture):
    """Radial flow architecture.

    Note: this model currently supports only one-way transformations.

    Reference: Rezende and Mohamed "Variational Inference with Normalizing Flows" (2016); https://arxiv.org/abs/1505.05770.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 **kwargs):
        super().__init__(event_shape, Radial, **kwargs)


class SylvesterFlow(ResidualArchitecture):
    """Sylvester flow architecture.

    Note: this model currently supports only one-way transformations.

    Reference: Van den Berg et al. "Sylvester Normalizing Flows for Variational Inference" (2019); https://arxiv.org/abs/1803.05649.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 **kwargs):
        super().__init__(event_shape, Sylvester, **kwargs)
