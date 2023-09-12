from typing import Union, Tuple

import torch

from normalizing_flows.bijections.continuous.base import ContinuousFlow, create_nn, ODEFunction


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(ContinuousFlow):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        diff_eq = ODEFunction(create_nn(self.n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
