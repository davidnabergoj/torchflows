from typing import Union, Tuple
import torch
from torchflows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from torchflows.bijections.finite.autoregressive.transformers.linear.matrix import LUTransformer
from torchflows.utils import get_batch_shape


class Invertible1x1ConvolutionTransformer(TensorTransformer):
    """
    Invertible 1x1 convolution.

    This transformer receives as input a batch of images x with x.shape (*batch_shape, channels, *image_dimensions) and
     parameters h for an invertible linear transform of the channels
     with h.shape = (*batch_shape, *parameter_shape).
    Note that image_dimensions can be a shape with arbitrarily ordered dimensions (height, width).
    In fact, it is not required that the image is two-dimensional. Voxels with shape (channels, height, width, depth)
    are also supported, as well as tensors with more general shapes.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        self.n_channels, *self.image_dimensions = event_shape
        self.invertible_linear: TensorTransformer = LUTransformer(event_shape=(self.n_channels,))

    @property
    def parameter_shape(self) -> Union[torch.Size, Tuple[int, ...]]:
        return self.invertible_linear.parameter_shape

    @property
    def default_parameters(self) -> torch.Tensor:
        return self.invertible_linear.default_parameters

    def apply_linear(self, inputs: torch.Tensor, h: torch.Tensor, forward: bool):
        batch_shape = get_batch_shape(inputs, self.event_shape)

        n_batch_dims = len(batch_shape)
        n_image_dims = len(self.image_dimensions)

        # (*batch_shape, n_channels, *image_dims) -> (*image_dims, *batch_shape, n_channels)
        inputs = torch.permute(
            inputs,
            (
                *list(range(n_batch_dims + 1, n_batch_dims + 1 + n_image_dims)),  # image_dims is moved to the start
                *list(range(n_batch_dims)),  # batch_shape is moved to the middle
                n_batch_dims  # n_channels is moved to the end
            )
        )

        # Apply linear transformation along channel dimension
        if forward:
            outputs, log_det = self.invertible_linear.forward(inputs, h)
        else:
            outputs, log_det = self.invertible_linear.inverse(inputs, h)
        # outputs and log_det need to be permuted now.

        outputs = torch.permute(
            outputs,
            (
                *list(range(n_image_dims, n_image_dims + n_batch_dims)),  # batch_shape is moved to the start
                n_image_dims + n_batch_dims,  # n_channels is moved to the middle,
                *list(range(n_image_dims)),  # image_dims is moved to the end
            )
        )
        return outputs, log_det

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_linear(x, h, forward=True)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_linear(z, h, forward=False)
