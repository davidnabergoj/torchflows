from typing import Union, Tuple

import torch

from normalizing_flows.bijections.continuous.base import ApproximateContinuousBijection, \
    RegularizedApproximateODEFunction, create_nn_time_independent


class DeepDiffeomorphicBijection(ApproximateContinuousBijection):
    """
    Base bijection for the DDNF model.
    Note that this model is implemented WITHOUT Geodesic regularization.
    This is because torchdiffeq ODE solvers do not output the predicted velocity, only the point.
    While the paper presents DDNF as a continuous normalizing flow, it is easier implement as a Residual normalizing
        flow in this library.

    IMPORTANT NOTE: the Euler solver prouduces very inaccurate results. Switching to the DOPRI5 solver massively
    improves reconstruction quality. However, we leave the Euler solver as it is presented in the original method.

    Salman et al. Deep diffeomorphic normalizing flows (2018).
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], n_steps: int = 150, **kwargs):
        """
        :param n_steps: parameter T in the paper, i.e. the number of ResNet cells.
        """
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        diff_eq = RegularizedApproximateODEFunction(create_nn_time_independent(n_dim))
        self.n_steps = n_steps
        super().__init__(event_shape, diff_eq, solver="euler", **kwargs)  # USE DOPRI5 for stability
