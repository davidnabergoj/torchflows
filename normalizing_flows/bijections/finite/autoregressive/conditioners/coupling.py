import torch

from normalizing_flows.bijections.finite.autoregressive.conditioners.base import Conditioner
from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform
from normalizing_flows.utils import get_batch_shape


class Coupling(Conditioner):
    def __init__(self, event_shape: torch.Size, constants: torch.Tensor):
        """
        Coupling conditioner.


        Note: Always treats the first n_dim // 2 dimensions as constant.
        Shuffling is handled in Permutation bijections.

        :param constants:
        """
        super().__init__()
        self.event_shape = event_shape

        # TODO add support for other kinds of masks
        n_constant_dims = int(torch.prod(torch.tensor(event_shape)))
        self.constant_mask = torch.less(torch.arange(n_constant_dims).view(*event_shape), (n_constant_dims // 2))
        self.register_buffer('constants', constants)  # Takes care of torch devices

    @property
    @torch.no_grad()
    def input_shape(self):
        return (int(torch.sum(self.constant_mask)),)

    @property
    @torch.no_grad()
    def output_shape(self):
        return (int(torch.sum(~self.constant_mask)),)

    def forward(self, x: torch.Tensor, transform: ConditionerTransform, context: torch.Tensor = None) -> torch.Tensor:
        # Predict transformer parameters for output dimensions
        batch_shape = get_batch_shape(x, self.event_shape)
        x_const = x.view(*batch_shape, *self.event_shape)[..., self.constant_mask]
        tmp = transform(x_const, context=context)
        n_parameters = tmp.shape[-1]

        # Create full parameter tensor
        h = torch.empty(size=(*batch_shape, *self.event_shape, n_parameters), dtype=x.dtype, device=x.device)

        # Fill the parameter tensor with predicted values
        h[..., ~self.constant_mask, :] = tmp
        h[..., self.constant_mask, :] = self.constants

        return h
