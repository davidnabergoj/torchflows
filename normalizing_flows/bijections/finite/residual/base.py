from typing import Union, Tuple, List

import torch

from normalizing_flows.bijections.base import Bijection, BijectiveComposition
from normalizing_flows.utils import get_batch_shape, unflatten_event, flatten_event


class ResidualBijection(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        """

        g maps from (*batch_shape, n_event_dims) to (*batch_shape, n_event_dims)

        :param event_shape:
        """
        super().__init__(event_shape)
        self.g: callable = None

    def log_det(self, x, **kwargs):
        raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        z = x + unflatten_event(self.g(flatten_event(x, self.event_shape)), self.event_shape)

        if skip_log_det:
            log_det = torch.full(size=batch_shape, fill_value=torch.nan)
        else:
            x_flat = flatten_event(x, self.event_shape).clone()
            x_flat.requires_grad_(True)
            log_det = self.log_det(x_flat, training=self.training)

        return z, log_det

    def inverse(self,
                z: torch.Tensor,
                context: torch.Tensor = None,
                skip_log_det: bool = False,
                n_iterations: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        x = z
        for _ in range(n_iterations):
            x = z - unflatten_event(self.g(flatten_event(x, self.event_shape)), self.event_shape)

        if skip_log_det:
            log_det = torch.full(size=batch_shape, fill_value=torch.nan)
        else:
            x_flat = flatten_event(x, self.event_shape).clone()
            x_flat.requires_grad_(True)
            log_det = -self.log_det(x_flat, training=self.training)

        return x, log_det


class ResidualComposition(BijectiveComposition):
    def __init__(self, blocks: List[ResidualBijection]):
        assert len(blocks) > 0
        super().__init__(
            event_shape=blocks[0].event_shape,
            layers=blocks,
            context_shape=blocks[0].context_shape
        )
