from typing import Union, Tuple

import torch

from normalizing_flows.bijections.continuous.base import ContinuousBijection, create_nn, RegularizedODEFunction


# https://github.com/cfinlay/ffjord-rnode/blob/master/train.py

class RNode(ContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = RegularizedODEFunction(create_nn(n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
