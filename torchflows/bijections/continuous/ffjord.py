from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ApproximateContinuousBijection,
    create_nn,
    RegularizedApproximateODEFunction,
    create_cnn
)


# https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py

class FFJORD(ApproximateContinuousBijection):
    """Free-form Jacobian of reversible dynamics (FFJORD) architecture.

    Gratwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (2018); https://arxiv.org/abs/1810.01367.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], nn_kwargs: dict = None, **kwargs):
        nn_kwargs = nn_kwargs or {}
        diff_eq = RegularizedApproximateODEFunction(create_nn(event_shape, **nn_kwargs))
        super().__init__(event_shape, diff_eq, solver='rk4', **kwargs)


class ConvolutionalFFJORD(ApproximateContinuousBijection):
    """Convolutional variant of the FFJORD architecture.

    Gratwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (2018); https://arxiv.org/abs/1810.01367.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], nn_kwargs: dict = None, **kwargs):
        nn_kwargs = nn_kwargs or {}
        if len(event_shape) != 3:
            raise ValueError("Event shape must be of length 3 (channels, height, width).")
        diff_eq = RegularizedApproximateODEFunction(create_cnn(event_shape[0], **nn_kwargs))
        super().__init__(event_shape, diff_eq, solver='rk4', **kwargs)
