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

    def integral(self, x, h) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, h: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO recieve h as a torch.Tensor, not a list of tensors
        with torch.enable_grad():
            x = x.requires_grad_()
            z = self.integral(x, h)
        jac = torch.autograd.grad(z, x, torch.ones_like(z), create_graph=True)[0]
        log_det = jac.log()
        return z, log_det

    def inverse(self, z: torch.Tensor, h: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = bisection(
            f=self.integral,
            y=z,
            a=torch.full_like(z, -self.bound),
            b=torch.full_like(z, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            h=h
        )
        return x, -self.forward(x, h)[1]
