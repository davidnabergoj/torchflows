from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ApproximateContinuousBijection,
    RegularizedApproximateODEFunction,
    create_nn_time_independent,
    create_cnn_time_independent
)


class DeepDiffeomorphicBijection(ApproximateContinuousBijection):
    """Deep diffeomorphic normalizing flow (DDNF) architecture.

    Notes:
        - this model is implemented without Geodesic regularization. This is because torchdiffeq ODE solvers do not output the predicted velocity, only the point.
        - while the paper presents DDNF as a continuous normalizing flow, it implemented as a residual normalizing flow in this library. There is no functional difference.
        - IMPORTANT: the Euler solver produces very inaccurate results. Switching to the DOPRI5 solver massively improves reconstruction quality. However, we leave the Euler solver as it is presented in the original method.

    Reference: Salman et al. "Deep diffeomorphic normalizing flows" (2018); https://arxiv.org/abs/1810.03256.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_steps: int = 150,
                 solver="euler",
                 nn_kwargs: dict = None,
                 **kwargs):
        """
        Constructor.

        :param event_shape: shape of the event tensor.
        :param n_steps: parameter T in the paper, i.e. the number of ResNet cells.
        """
        nn_kwargs = nn_kwargs or {}
        diff_eq = RegularizedApproximateODEFunction(create_nn_time_independent(event_shape, **nn_kwargs))
        self.n_steps = n_steps
        super().__init__(event_shape, diff_eq, solver=solver, **kwargs)


class ConvolutionalDeepDiffeomorphicBijection(ApproximateContinuousBijection):
    """Convolutional variant of the DDNF architecture.

    Reference: Salman et al. "Deep diffeomorphic normalizing flows" (2018); https://arxiv.org/abs/1810.03256.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_steps: int = 150,
                 solver="euler",
                 nn_kwargs: dict = None,
                 **kwargs):
        nn_kwargs = nn_kwargs or {}
        if len(event_shape) != 3:
            raise ValueError("Event shape must be of length 3 (channels, height, width).")
        diff_eq = RegularizedApproximateODEFunction(create_cnn_time_independent(event_shape[0], **nn_kwargs))
        self.n_steps = n_steps
        super().__init__(event_shape, diff_eq, solver=solver, **kwargs)
