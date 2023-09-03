from typing import Union, Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.base import MonotonicSpline


class Cubic(MonotonicSpline):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
