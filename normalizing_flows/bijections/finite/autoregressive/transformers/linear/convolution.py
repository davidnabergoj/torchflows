from typing import Union, Tuple
import torch

from normalizing_flows.bijections import LU
from normalizing_flows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from normalizing_flows.utils import sum_except_batch, get_batch_shape


class Invertible1x1Convolution(TensorTransformer):
    """
    Invertible 1x1 convolution.

    This transformer receives as input a batch of images x with x.shape (*batch_shape, *image_dimensions, channels) and
     parameters h for an invertible linear transform of the channels
     with h.shape = (*batch_shape, *image_dimensions, *parameter_shape).
    Note that image_dimensions can be a shape with arbitrarily ordered dimensions (height, width).
    In fact, it is not required that the image is two-dimensional. Voxels with shape (height, width, depth, channels)
    are also supported, as well as tensors with more general shapes.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)
        *self.image_dimensions, self.n_channels = event_shape
        self.invertible_linear: TensorTransformer = LU(event_shape=(self.n_channels,))

    @property
    def parameter_shape(self) -> Union[torch.Size, Tuple[int, ...]]:
        return self.invertible_linear.parameter_shape

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=self.parameter_shape)

    def apply_linear(self, inputs: torch.Tensor, h: torch.Tensor, forward: bool):
        batch_shape = get_batch_shape(inputs, self.event_shape)

        h = h / self.const + self.default_parameters[[None] * len(batch_shape)]
        h_flat = torch.flatten(h, start_dim=0, end_dim=len(batch_shape) + len(self.image_dimensions))
        inputs_flat = torch.flatten(inputs, start_dim=0, end_dim=len(batch_shape) + len(self.image_dimensions))

        # Apply linear transformation along channel dimension
        if forward:
            outputs_flat, log_det_flat = self.invertible_linear.forward(inputs_flat, h_flat)
        else:
            outputs_flat, log_det_flat = self.invertible_linear.inverse(inputs_flat, h_flat)
        outputs = outputs_flat.view_as(inputs)
        log_det = sum_except_batch(
            log_det_flat.view(*batch_shape, *self.image_dimensions),
            event_shape=self.image_dimensions
        )
        return outputs, log_det

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_linear(x, h, forward=True)

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_linear(z, h, forward=False)
