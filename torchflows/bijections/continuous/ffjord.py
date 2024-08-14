from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ApproximateContinuousBijection,
    create_nn,
    RegularizedApproximateODEFunction
)


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(ApproximateContinuousBijection):
    """ Free-form Jacobian of reversible dynamics (FFJORD) architecture.

    Gratwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (2018); https://arxiv.org/abs/1810.01367.
    """
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = RegularizedApproximateODEFunction(create_nn(n_dim))
        super().__init__(event_shape, diff_eq, **kwargs)
