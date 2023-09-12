from typing import Union, Tuple

import torch

from normalizing_flows.bijections.continuous.base import ContinuousBijection, create_nn, ODEFunctionBasic


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(ContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = ODEFunctionBasic(create_nn(n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
