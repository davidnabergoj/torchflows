from typing import Union, Tuple

import torch

from torchflows.bijections.base import Bijection
from torchflows.utils import get_batch_shape


class InvertibleMatrix(Bijection):
    """
    Invertible matrix bijection (currently ignores context).
    """

    def __init__(self, 
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 l2_regularization: bool = False,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        self.l2_regularization = l2_regularization
        self.register_buffer('device_buffer', torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        x_flat = x.view(*batch_shape, -1)
        context_flat = context.view(*batch_shape, -1) if context is not None else None
        z_flat = self.project_flat(x_flat, context_flat)
        z = z_flat.view_as(x)
        log_det = self.log_det_project()[[None] * len(batch_shape)].repeat(*batch_shape, 1).squeeze(-1)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        z_flat = z.view(*batch_shape, -1)
        context_flat = context.view(*batch_shape, -1) if context is not None else None
        x_flat = self.solve_flat(z_flat, context_flat)
        x = x_flat.view_as(z)
        log_det = -self.log_det_project()[[None] * len(batch_shape)].repeat(*batch_shape, 1).squeeze(-1)
        return x, log_det

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Find x in Ax = b where b is given and A is this matrix.

        :param b_flat: shift tensor with shape (self.n_dim,)
        :param context:
        :return:
        """
        raise NotImplementedError

    def log_det_project(self) -> torch.Tensor:
        """

        :return: log abs det jac of f where f(x) = Ax and A is this matrix.
        """
        raise NotImplementedError

    def regularization(self, *aux):
        """Compute regularization.

        :param Tuple[Any, ...] aux: unused.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`. 
        """
        if self.l2_regularization:
            return sum([
                torch.sum(torch.square(p)) 
                for p in self.parameters()
                if p.requires_grad
            ])
        else:
            return torch.tensor(0.0)