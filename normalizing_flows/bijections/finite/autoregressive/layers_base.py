from typing import Tuple, Optional

import torch

from normalizing_flows.bijections.finite.autoregressive.conditioners.base import Conditioner, NullConditioner
from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform, Constant
from normalizing_flows.bijections.finite.autoregressive.conditioners.coupling_masks import CouplingMask
from normalizing_flows.bijections.finite.autoregressive.transformers.base import TensorTransformer, ScalarTransformer
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.utils import flatten_event, unflatten_event, get_batch_shape


class AutoregressiveBijection(Bijection):
    def __init__(self,
                 event_shape,
                 conditioner: Optional[Conditioner],
                 transformer: TensorTransformer,
                 conditioner_transform: ConditionerTransform,
                 **kwargs):
        super().__init__(event_shape=event_shape)
        self.conditioner = conditioner
        self.conditioner_transform = conditioner_transform
        self.transformer = transformer

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(x, transform=self.conditioner_transform, context=context)
        z, log_det = self.transformer(x, h)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conditioner(z, transform=self.conditioner_transform, context=context)
        x, log_det = self.transformer.inverse(z, h)
        return x, log_det


class CouplingBijection(AutoregressiveBijection):
    """
    Base coupling bijection object.

    A coupling bijection is defined using a transformer, conditioner transform, and always a coupling conditioner.

    The coupling conditioner receives as input an event tensor x.
    It then partitions an input event tensor x into a constant part x_A and a modifiable part x_B.
    For x_A, the conditioner outputs a set of parameters which is always the same.
    For x_B, the conditioner outputs a set of parameters which are predicted from x_A.

    Coupling conditioners differ in the partitioning method. By default, the event is flattened; the first half is x_A
     and the second half is x_B. When using this in a normalizing flow, permutation layers can shuffle event dimensions.

    For improved performance, this implementation does not use a standalone coupling conditioner. It instead implements
     a method to partition x into x_A and x_B and then predict parameters for x_B.
    """

    def __init__(self,
                 transformer: TensorTransformer,
                 coupling_mask: CouplingMask,
                 conditioner_transform: ConditionerTransform,
                 **kwargs):
        super().__init__(coupling_mask.event_shape, None, transformer, conditioner_transform, **kwargs)
        self.coupling_mask = coupling_mask

        assert conditioner_transform.input_event_shape == (coupling_mask.constant_event_size,)
        assert transformer.event_shape == (self.coupling_mask.transformed_event_size,)

    def partition_and_predict_parameters(self, x: torch.Tensor, context: torch.Tensor):
        """
        Partition tensor x and compute transformer parameters.

        :param x: input tensor with x.shape = (*batch_shape, *event_shape) to be partitioned into x_A and x_B.
        :param context: context tensor with context.shape = (*batch_shape, *context.shape).
        :return: parameter tensor h with h.shape = (*batch_shape, *parameter_shape).
        """
        # Predict transformer parameters for output dimensions
        x_a = x[..., self.coupling_mask.mask]  # (*b, constant_event_size)
        h_b = self.conditioner_transform(x_a, context=context)  # (*b, *p)
        return h_b

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x.clone()
        h_b = self.partition_and_predict_parameters(x, context)
        z[..., ~self.coupling_mask.mask], log_det = self.transformer.forward(x[..., ~self.coupling_mask.mask], h_b)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z.clone()
        h_b = self.partition_and_predict_parameters(x, context)
        x[..., ~self.coupling_mask.mask], log_det = self.transformer.inverse(z[..., ~self.coupling_mask.mask], h_b)
        return x, log_det


class ForwardMaskedAutoregressiveBijection(AutoregressiveBijection):
    def __init__(self, conditioner: Conditioner, transformer: ScalarTransformer,
                 conditioner_transform: ConditionerTransform):
        super().__init__(conditioner, transformer, conditioner_transform)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        n_event_dims = int(torch.prod(torch.as_tensor(self.event_shape)))
        log_det = torch.zeros(size=batch_shape, device=z.device)
        x_flat = flatten_event(torch.clone(z), self.event_shape)
        for i in torch.arange(n_event_dims):
            x_clone = unflatten_event(torch.clone(x_flat), self.event_shape)
            h = self.conditioner(
                x_clone,
                transform=self.conditioner_transform,
                context=context
            )
            tmp, log_det = self.transformer.inverse(x_clone, h)
            x_flat[..., i] = flatten_event(tmp, self.event_shape)[..., i]
        x = unflatten_event(x_flat, self.event_shape)
        return x, log_det


class InverseMaskedAutoregressiveBijection(AutoregressiveBijection):
    def __init__(self, conditioner: Conditioner, transformer: ScalarTransformer,
                 conditioner_transform: ConditionerTransform):
        super().__init__(conditioner, transformer, conditioner_transform)
        self.forward_layer = ForwardMaskedAutoregressiveBijection(
            conditioner,
            transformer,
            conditioner_transform
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_layer.inverse(x, context=context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_layer.forward(z, context=context)


class ElementwiseBijection(AutoregressiveBijection):
    """
    Base elementwise bijection class.

    Applies a bijective transformation to each element of the input tensor.
    The bijection for each element has its own set of globally learned parameters.
    """

    def __init__(self, transformer: ScalarTransformer, fill_value: float = None):
        super().__init__(
            transformer.event_shape,
            NullConditioner(),
            transformer,
            Constant(transformer.event_shape, transformer.parameter_shape, fill_value=fill_value)
        )
