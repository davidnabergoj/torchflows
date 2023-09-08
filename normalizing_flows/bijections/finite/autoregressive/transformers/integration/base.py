import math
from typing import Union, Tuple, List

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.util import bisection


class Integration(Transformer):
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

    def base_forward(self, x: torch.Tensor, params: List[torch.Tensor]):
        with torch.enable_grad():
            x = x.requires_grad_()
            z = self.integral(x, params)
        jac = torch.autograd.grad(z, x, torch.ones_like(z), create_graph=True)[0]
        log_det = jac.log()
        return z, log_det

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.compute_parameters(h)
        return self.base_forward(x, params)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.compute_parameters(h)
        x = bisection(
            f=self.integral,
            y=z,
            a=torch.full_like(z, -self.bound),
            b=torch.full_like(z, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            h=params
        )
        return x, -self.base_forward(x, params)[1]
