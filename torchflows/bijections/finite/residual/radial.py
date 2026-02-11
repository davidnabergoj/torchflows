from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softplus

from torchflows.bijections.finite.residual.base import ClassicResidualBijection
from torchflows.utils import get_batch_shape


class Radial(ClassicResidualBijection):
    """Implements the Radial bijection layer. Internally flattens events. Constrains alpha with softplus.
    
    Reference: Rezende and Mohamed "Variational Inference with Normalizing Flows" (2016); https://arxiv.org/abs/1505.05770.
    """
    # as per Rezende, Mohamed (2015)

    def __init__(self, 
                 event_shape: Union[torch.Size, Tuple[int, ...]], 
                 **kwargs):
        """Radial constructor.
        
        :param event_shape: shape of the event tensor.
        :param kwargs: keyword arguments for `ClassicResidualBijection`.
        """
        super().__init__(event_shape, **kwargs)
        self.beta = nn.Parameter(torch.randn(size=()) / self.n_dim)
        self.unconstrained_alpha = nn.Parameter(torch.randn(size=()) / self.n_dim)
        self.z0 = nn.Parameter(torch.randn(size=(self.n_dim,)) / self.n_dim)

        self.eps = 1e-8

    @property
    def alpha(self):
        """Computes alpha, a positive parameter for the radial bijection."""
        return softplus(self.unconstrained_alpha)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the inverse radial transformation.
        
        :param torch.Tensor z: tensor with shape `(*batch_shape, *event_shape)`.
        :param Optional[torch.Tensor] context: context tensor with shape `(*batch_shape, *context_shape)`.
            Currently unused.
        :rtype: Tuple[torch.Tensor, torch.Tensor].
        :return: transformed tensor with shape `(*batch_shape, *event_shape)` and the associated log Jacobian 
            determinant with shape `batch_shape`.
        """
        # Flatten event
        batch_shape = get_batch_shape(z, self.event_shape)
        z = z.view(*batch_shape, self.n_dim)

        # Compute auxiliary variables
        z0 = self.z0.view(*([1] * len(batch_shape)), *self.z0.shape)
        r = torch.abs(z - z0)
        h = 1 / (self.alpha + r + self.eps)

        # Compute transformed point
        x = z + self.beta * h * (z - z0)

        # Compute determinant of the Jacobian
        log_det = torch.add(
            torch.log1p((self.beta * self.alpha / ((self.alpha + r) ** 2 + self.eps))),
            torch.log1p(self.beta * h) * (self.n_dim - 1)
        ).sum(dim=-1)

        # Unflatten event
        x = x.view(*batch_shape, *self.event_shape)
        return x, log_det
