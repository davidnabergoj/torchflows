from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ApproximateContinuousBijection,
    create_nn,
    RegularizedApproximateODEFunction,
    create_cnn
)


# https://github.com/cfinlay/ffjord-rnode/blob/master/train.py

class RNODE(ApproximateContinuousBijection):
    """Regularized neural ordinary differential equation (RNODE) architecture.

    Reference: Finlay et al. "How to train your neural ODE: the world of Jacobian and kinetic regularization" (2020); https://arxiv.org/abs/2002.02798.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        diff_eq = RegularizedApproximateODEFunction(create_nn(event_shape, hidden_size=100, n_hidden_layers=1),
                                                    regularization="sq_jac_norm")
        super().__init__(event_shape, diff_eq, **kwargs)


class ConvolutionalRNODE(ApproximateContinuousBijection):
    """Convolutional variant of the RNODE architecture

    Reference: Finlay et al. "How to train your neural ODE: the world of Jacobian and kinetic regularization" (2020); https://arxiv.org/abs/2002.02798.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        if len(event_shape) != 3:
            raise ValueError("Event shape must be of length 3 (channels, height, width).")
        diff_eq = RegularizedApproximateODEFunction(create_cnn(event_shape[0]), regularization="sq_jac_norm")
        super().__init__(event_shape, diff_eq, **kwargs)
