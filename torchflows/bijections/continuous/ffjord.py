from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ApproximateContinuousBijection,
    create_nn,
    RegularizedApproximateODEFunction
)


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(ApproximateContinuousBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = RegularizedApproximateODEFunction(create_nn(n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
