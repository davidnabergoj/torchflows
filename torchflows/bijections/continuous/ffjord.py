from typing import Union, Tuple

import torch

from torchflows.bijections.continuous.base import (
    ContinuousBijection,
    HutchinsonTimeDerivative,
    create_dnn_forward_model,
    create_cnn
)


class FFJORD(ContinuousBijection):
    """FFJORD architecture for general tensors. 
    Parameterizes the time derivative with a feed-forward neural network.

    Gratwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (2018).
    URL: https://arxiv.org/abs/1810.01367.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """FFJORD constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param dict nn_kwargs: keyword arguments for `create_nn`.
        :param dict time_derivative_kwargs: keyword arguments for `ApproximateTimeDerivative`.
        :param kwargs: keyword arguments for `ContinuousBijection`.
        """
        time_derivative_kwargs = time_derivative_kwargs or {}
        time_derivative_kwargs.update(reg_jac=True)
        super().__init__(
            event_shape=event_shape,
            f=HutchinsonTimeDerivative(
                create_dnn_forward_model(event_shape, **(nn_kwargs or {})),
                **time_derivative_kwargs
            ),
            **kwargs
        )


class ConvolutionalFFJORD(ContinuousBijection):
    """FFJORD architecture for images. 
    Parameterizes the time derivative with a convolutional neural network.
       
    Gratwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (2018); 
    URL: https://arxiv.org/abs/1810.01367.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """ConvolutionalFFJORD constructor.

        :param Union[Tuple[int, ...], torch.Size] event_shape: shape of the event tensor.
        :param dict nn_kwargs: keyword arguments for `create_cnn`.
        :param dict time_derivative_kwargs: keyword arguments for `ApproximateTimeDerivative`.
        :param kwargs: keyword arguments for `ContinuousBijection`.
        """
        if len(event_shape) != 3:
            raise ValueError(
                "Event shape must be of length 3 (channels, height, width)."
            )
        
        time_derivative_kwargs = time_derivative_kwargs or {}
        time_derivative_kwargs.update(reg_jac=True)
        super().__init__(
            event_shape=event_shape,
            f=HutchinsonTimeDerivative(
                create_cnn(event_shape[0], **(nn_kwargs or {})),
                **time_derivative_kwargs
            ),
            **kwargs
        )
