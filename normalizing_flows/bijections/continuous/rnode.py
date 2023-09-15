from typing import Union, Tuple

import torch

from normalizing_flows.bijections.continuous.base import ApproximateContinuousBijection, create_nn, RegularizedApproximateODEFunction


# https://github.com/cfinlay/ffjord-rnode/blob/master/train.py

class RNODE(ApproximateContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = RegularizedApproximateODEFunction(create_nn(n_dim), regularization="sq_jac_norm")
        super().__init__(event_shape, diff_eq, **kwargs)
