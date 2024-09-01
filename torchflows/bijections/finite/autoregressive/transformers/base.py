from typing import Tuple, Union

import torch

from torchflows.bijections.base import Bijection


class TensorTransformer(Bijection):
    """
    Base transformer class.

    A transformer receives as input a tensor x with x.shape = (*batch_shape, *event_shape) and parameters h
    with h.shape = (*batch_shape, *parameter_shape). It applies a bijective map to each tensor in the batch
    with its corresponding parameter set. In general, the parameters are used to transform the entire tensor at
    once. As a special case, the subclass ScalarTransformer transforms each element of an input event
    individually.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape=event_shape)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the forward transformation.

        :param torch.Tensor x: input tensor with shape (*batch_shape, *event_shape).
        :param torch.Tensor h: parameter tensor with shape (*batch_shape, *parameter_shape).
        :returns: output tensor with shape (*batch_shape, *event_shape).
        """
        raise NotImplementedError

    def inverse(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the inverse transformation.

        :param torch.Tensor x: input tensor with shape (*batch_shape, *event_shape).
        :param torch.Tensor h: parameter tensor with shape (*batch_shape, *parameter_shape).
        :returns: output tensor with shape (*batch_shape, *event_shape).
        """
        raise NotImplementedError

    @property
    def parameter_shape(self) -> Union[torch.Size, Tuple[int, ...]]:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return int(torch.prod(torch.as_tensor(self.parameter_shape)))

    @property
    def default_parameters(self) -> torch.Tensor:
        """
        Set of parameters which ensures an identity transformation.
        """
        raise NotImplementedError


class ScalarTransformer(TensorTransformer):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    @property
    def parameter_shape_per_element(self):
        """
        The shape of parameters that transform a single element of an input tensor.

        Example:
            * if using an affine transformer, this is equal to (2,) (corresponding to scale and shift).
            * if using a rational quadratic spline transformer, this is equal to (3 * b - 1,) where b is the
              number of bins.
        """
        raise NotImplementedError

    @property
    def n_parameters_per_element(self):
        return int(torch.prod(torch.as_tensor(self.parameter_shape_per_element)))

    @property
    def parameter_shape(self) -> Union[torch.Size, Tuple[int, ...]]:
        # Scalar transformers map each element individually, so the first dimensions are the event shape
        return torch.Size((*self.event_shape, *self.parameter_shape_per_element))
