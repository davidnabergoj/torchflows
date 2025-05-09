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

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], nn_kwargs: dict = None, **kwargs):
        default_nn_kwargs = {'hidden_size': 100, 'n_hidden_layers': 1}
        nn_kwargs = nn_kwargs or dict()
        default_nn_kwargs.update(nn_kwargs)
        diff_eq = RegularizedApproximateODEFunction(
            create_nn(event_shape, **default_nn_kwargs),
            regularization="sq_jac_norm"
        )
        super().__init__(event_shape, diff_eq, solver='rk4', **kwargs)


class ConvolutionalRNODE(ApproximateContinuousBijection):
    """Convolutional variant of the RNODE architecture

    Reference: Finlay et al. "How to train your neural ODE: the world of Jacobian and kinetic regularization" (2020); https://arxiv.org/abs/2002.02798.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], nn_kwargs: dict = None, **kwargs):
        default_nn_kwargs = {'n_layers': 2}
        nn_kwargs = nn_kwargs or dict()
        default_nn_kwargs.update(nn_kwargs)
        if len(event_shape) != 3:
            raise ValueError("Event shape must be of length 3 (channels, height, width).")
        diff_eq = RegularizedApproximateODEFunction(
            create_cnn(event_shape[0], **default_nn_kwargs),
            regularization="sq_jac_norm"
        )
        super().__init__(event_shape, diff_eq, solver='rk4', **kwargs)
