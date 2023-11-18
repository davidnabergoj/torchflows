from typing import Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.conditioners.base import Conditioner, NullConditioner
from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform, Constant
from normalizing_flows.bijections.finite.autoregressive.conditioners.coupling import Coupling
from normalizing_flows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.utils import flatten_event, unflatten_event, get_batch_shape


class AutoregressiveBijection(Bijection):
    def __init__(self, conditioner: Conditioner, transformer: ScalarTransformer, conditioner_transform: ConditionerTransform):
        super().__init__(event_shape=transformer.event_shape)
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
    def __init__(self, conditioner: Coupling, transformer: ScalarTransformer, conditioner_transform: ConditionerTransform,
                 **kwargs):
        super().__init__(conditioner, transformer, conditioner_transform, **kwargs)

        # We need to change the transformer event shape because it will no longer accept full-shaped events, but only
        # a flattened selection of event dimensions.
        self.transformer.event_shape = torch.Size((self.conditioner.n_changed_dims,))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x.clone()
        h, mask = self.conditioner(x, self.conditioner_transform, context, return_mask=True)
        z[..., ~mask], log_det = self.transformer.forward(x[..., ~mask], h[..., ~mask, :])
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z.clone()
        h, mask = self.conditioner(z, self.conditioner_transform, context, return_mask=True)
        x[..., ~mask], log_det = self.transformer.inverse(z[..., ~mask], h[..., ~mask, :])
        return x, log_det


class ForwardMaskedAutoregressiveBijection(AutoregressiveBijection):
    def __init__(self, conditioner: Conditioner, transformer: ScalarTransformer, conditioner_transform: ConditionerTransform):
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
    def __init__(self, conditioner: Conditioner, transformer: ScalarTransformer, conditioner_transform: ConditionerTransform):
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
    def __init__(self, transformer: ScalarTransformer, n_transformer_parameters: int):
        super().__init__(NullConditioner(), transformer, Constant(transformer.event_shape, n_transformer_parameters))
        # TODO override forward and inverse to save on space
