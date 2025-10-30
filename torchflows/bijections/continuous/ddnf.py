from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ContinuousBijection,
    HutchinsonTimeDerivative,
    create_cnn,
    create_dnn_forward_model,
)


class DDNF(ContinuousBijection):
    """DDNF architecture for general tensors.
    Parameterizes the time derivative with a feed-forward neural network.

    TODO: implement Geodesic regularization!

    Note: The Euler solver produces very inaccurate results. Switching to the DOPRI5 solver massively improves 
        reconstruction quality. We leave the Euler solver as it is presented in the original method.

    Salman et al. "Deep diffeomorphic normalizing flows" (2018).
    URL: https://arxiv.org/abs/1810.03256.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_steps: int = 150,
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """DeepDiffeomorphicBijection constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param n_steps: number of ResNet cells (Euler integration steps). Parameter T in the paper.
        :param dict nn_kwargs: keyword arguments for `create_nn`.
        :param dict time_derivative_kwargs: keyword arguments for `ApproximateTimeDerivative`.
        :param kwargs: keyword arguments for `ContinuousBijection`.
        """
        if 'solver' not in kwargs:
            kwargs['solver'] = 'euler'
        elif kwargs['solver'] != 'euler':
            raise ValueError("Only Euler solver permitted")
        
        # TODO: unify regularization in a single method instead of having all these separate architectures
        if 'reg_jac' in time_derivative_kwargs and time_derivative_kwargs['reg_jac']:
            raise ValueError("DDNF does not utilize Jacobian regularization")
        time_derivative_kwargs['reg_jac'] = False
        
        self.n_steps = n_steps
        time_derivative_kwargs = time_derivative_kwargs or {}
        # time_derivative_kwargs.update(reg_geo=True)  # TODO implement
        super().__init__(
            event_shape=event_shape,
            f=HutchinsonTimeDerivative(
                forward_model=create_dnn_forward_model(
                    event_shape, 
                    **(nn_kwargs or {})
                ),
                **time_derivative_kwargs
            ),
            **kwargs
        )

    def inverse(self, *args, **kwargs):
        # Set number of Euler integration steps for odeint call
        if 'options' not in kwargs:
            kwargs['options'] = {
                'step_size': 1 / self.n_steps
            }
        return super().inverse(*args, **kwargs)

class ConvolutionalDDNF(ContinuousBijection):
    """DDNF architecture for images.
    Parameterizes the time derivative with a convolutional neural network.

    TODO: implement Geodesic regularization!

    Note: The Euler solver can produce very inaccurate results. Switching to the DOPRI5 solver massively improves 
        reconstruction quality. We leave the Euler solver as it is presented in the original method.

    Salman et al. "Deep diffeomorphic normalizing flows" (2018).
    URL: https://arxiv.org/abs/1810.03256.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 n_steps: int = 150,
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """DeepDiffeomorphicBijection constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param n_steps: number of ResNet cells (Euler integration steps). Parameter T in the paper.
        :param dict nn_kwargs: keyword arguments for `create_nn`.
        :param dict time_derivative_kwargs: keyword arguments for `ApproximateTimeDerivative`.
        :param kwargs: keyword arguments for `ContinuousBijection`.
        """
        if len(event_shape) != 3:
            raise ValueError(
                "Event shape must be of length 3 (channels, height, width)."
            )

        if 'solver' not in kwargs:
            kwargs['solver'] = 'euler'
        elif kwargs['solver'] != 'euler':
            raise ValueError("Only Euler solver permitted")
        # TODO: unify regularization in a single method instead of having all these separate architectures
        if 'reg_jac' in time_derivative_kwargs and time_derivative_kwargs['reg_jac']:
            raise ValueError("DDNF does not utilize Jacobian regularization")
        time_derivative_kwargs['reg_jac'] = False

        self.n_steps = n_steps
        time_derivative_kwargs = time_derivative_kwargs or {}
        # time_derivative_kwargs.update(reg_geo=True)  # TODO implement
        super().__init__(
            event_shape=event_shape,
            f=HutchinsonTimeDerivative(
                create_cnn(event_shape[0], **(nn_kwargs or {})),
                **time_derivative_kwargs
            ),
            **kwargs
        )

    def inverse(self, *args, **kwargs):
        # Set number of Euler integration steps for odeint call
        if 'options' not in kwargs:
            kwargs['options'] = {
                'step_size': 1 / self.n_steps
            }
        return super().inverse(*args, **kwargs)
