import torch
import torch.nn as nn

from torchflows.utils import sum_except_batch


class ZeroCauchy(torch.distributions.Distribution, nn.Module):
    def __init__(self,
                 event_shape,
                 scale: float):
        super().__init__(event_shape=event_shape, validate_args=False)
        self.scale = scale
        self.dist = torch.distributions.Cauchy(loc=0.0, scale=self.scale)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.dist.sample(sample_shape=(*sample_shape, *self.event_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        elementwise_log_prob = self.dist.log_prob(value)
        return sum_except_batch(elementwise_log_prob, self.event_shape)
