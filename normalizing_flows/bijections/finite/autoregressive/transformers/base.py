from typing import Tuple, Union

import torch

from normalizing_flows.bijections.base import Bijection


class Transformer(Bijection):
    """
    Base transformer class.

    A transformer receives as input a batch of tensors x with x.shape = (*batch_shape, *event_shape) and a
     corresponding batch of parameter tensors h with h.shape = (*batch_shape, self.n_parameters). It outputs transformed
     tensors z with z.shape = (*batch_shape, *event_shape).
    Given parameters h, a transformer is bijective. Transformers apply bijections of the same kind to a batch of inputs,
     but use different parameters for each input.

    When implementing new transformers, consider the self.unflatten_conditioner_parameters method, which is used to
     optionally reshape transformer parameters into a suitable shape.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape=event_shape)

    def forward_base(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward bijection to a batch of inputs x, parameterizing each bijection with the corresponding parameter
         tensor in h.

        :param x: input tensor with x.shape = (*batch_shape, *event_shape).
        :param h: parameter tensor with h.shape = (*batch_shape, *parameter_shape).
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward bijection to a batch of inputs x, parameterizing each bijection with the corresponding parameter
         tensor in h.

        :param x: input tensor with x.shape = (*batch_shape, *event_shape).
        :param h: parameter tensor with h.shape = (*batch_shape, self.n_parameters).
        """
        return self.forward_base(x, self.unflatten_conditioner_parameters(h))

    def inverse_base(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse bijection to a batch of inputs x, parameterizing each bijection with the corresponding parameter
         tensor in h.

        :param x: input tensor with x.shape = (*batch_shape, *event_shape).
        :param h: parameter tensor with h.shape = (*batch_shape, *parameter_shape).
        """
        raise NotImplementedError

    def inverse(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse bijection to a batch of inputs x, parameterizing each bijection with the corresponding parameter
         tensor in h.

        :param x: input tensor with x.shape = (*batch_shape, *event_shape).
        :param h: parameter tensor with h.shape = (*batch_shape, self.n_parameters).
        """
        return self.inverse_base(x, self.unflatten_conditioner_parameters(h))

    def unflatten_conditioner_parameters(self, h: torch.Tensor):
        """
        Reshapes parameter tensors as predicted by the conditioner.
        The new shape facilitates operations in the transformer and facilitates transformer operations.
        If this method is not overwritten, the default parameter tensor shape is kept.

        :param h: batch of parameter tensors for each input event with shape (*batch_shape, self.n_parameters).
        :return: batch of parameter tensors for each input event with shape (*batch_shape, *new_shape)
        """
        return h

    @property
    def n_parameters(self) -> int:
        """
        Number of parameters that parametrize this transformer.

        Examples:
            * Rational quadratic splines require (3 * b - 1) * d where b is the number of bins and d is the
              dimensionality, equal to the product of all dimensions of the transformer input tensor.
            * An affine transformation requires 2 * d (typically corresponding to the unconstrained scale and shift).
        """
        raise NotImplementedError

    @property
    def default_parameters(self) -> torch.Tensor:
        """
        Set of parameters which yields a close-to-identity transformation.
        These are set to 0 by default.
        """
        return torch.zeros(size=(self.n_parameters,))
