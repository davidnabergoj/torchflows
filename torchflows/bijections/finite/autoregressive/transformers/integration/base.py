import math
from typing import Union, Tuple, List

import torch

from torchflows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from torchflows.bijections.numerical_inversion import bisection
from torchflows.utils import get_batch_shape, sum_except_batch


class Integration(ScalarTransformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], bound: float = 100.0, eps: float = 1e-6):
        """
        :param bound: specifies the initial interval [-bound, bound] where numerical inversion is performed.
        """
        super().__init__(event_shape)
        self.bound = bound
        self.eps = eps

    def integral(self, x, h: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def compute_parameters(self, h: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def base_forward_1d(self, x: torch.Tensor, params: List[torch.Tensor]):
        with torch.enable_grad():
            x = x.requires_grad_()
            z = self.integral(x, params)
        jac = torch.autograd.grad(z, x, torch.ones_like(z), create_graph=True)[0]
        log_det = jac.log()
        return z, log_det

    def forward_1d(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x.shape = (n,)
        h.shape = (n, n_parameters)
        """
        params = self.compute_parameters(h)
        return self.base_forward_1d(x, params)

    def inverse_1d_without_log_det(self, z: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        return bisection(
            f=self.integral,
            y=z,
            a=torch.full_like(z, -self.bound),
            b=torch.full_like(z, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            h=params
        )

    def inverse_1d(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z.shape = (n,)
        h.shape = (n, n_parameters)
        """
        params = self.compute_parameters(h)
        x = self.inverse_without_log_det(z, params)
        return x, -self.base_forward_1d(x, params)[1]

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x.shape = (*batch_shape, *event_shape)
        h.shape = (*batch_shape, *parameter_shape)
        """
        z_flat, log_det_flat = self.forward_1d(x.view(-1), h.view(-1, self.n_parameters_per_element))
        z = z_flat.view_as(x)
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = sum_except_batch(log_det_flat.view(*batch_shape, *self.event_shape), self.event_shape)
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat, log_det_flat = self.inverse_1d(z.view(-1), h.view(-1, self.n_parameters_per_element))
        x = x_flat.view_as(z)
        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = sum_except_batch(log_det_flat.view(*batch_shape, *self.event_shape), self.event_shape)
        return x, log_det
