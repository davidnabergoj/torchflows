from typing import Union, Tuple
import torch

from torchflows.bijections.continuous.base import (
    ContinuousBijection,
    HutchinsonTimeDerivative,
    create_cnn,
    create_dnn_forward_model
)


class RNODE(ContinuousBijection):
    """RNODE architecture for general tensors.
    Parameterizes the time derivative with a feed-forward neural network.

    TODO add kinetic regularization!

    Finlay et al. "How to train your neural ODE: the world of Jacobian and kinetic regularization" (2020).
    URL: https://arxiv.org/abs/2002.02798.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """RNODE constructor.

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
                event_shape=event_shape,
                forward_model=create_dnn_forward_model(
                    event_shape,
                    **(nn_kwargs or {})
                ),
                **time_derivative_kwargs
            ),
            **kwargs
        )

    def regularization(self, sq_jac_norm: torch.Tensor = None):
        """Compute Jacobian norm regularization.
        
        :param torch.Tensor sq_jac_norm: possible squared norm of the Jacobian. If provided, has shape `(batch_size,)`.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`.
        """
        if sq_jac_norm is not None:
            return self.f.reg_jac_coef * sq_jac_norm.mean()
        return torch.tensor(0.0)


class ConvolutionalRNODE(ContinuousBijection):
    """RNODE architecture for general tensors.
    Parameterizes the time derivative with a feed-forward neural network.

    TODO add kinetic regularization!

    Finlay et al. "How to train your neural ODE: the world of Jacobian and kinetic regularization" (2020).
    URL: https://arxiv.org/abs/2002.02798.
    """

    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 nn_kwargs: dict = None,
                 time_derivative_kwargs: dict = None,
                 **kwargs):
        """ConvolutionalRNODE constructor.

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


    def regularization(self, sq_jac_norm: torch.Tensor = None):
        """Compute Jacobian norm regularization.
        
        :param torch.Tensor sq_jac_norm: possible squared norm of the Jacobian. If provided, has shape `(batch_size,)`.
        :rtype: torch.Tensor.
        :return: regularization tensor with shape `()`.
        """
        if sq_jac_norm is not None:
            return self.f.reg_jac_coef * sq_jac_norm.mean()
        return torch.tensor(0.0)