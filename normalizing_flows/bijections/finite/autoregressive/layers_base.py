from typing import Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.conditioners.base import Conditioner, NullConditioner
from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform, Constant
from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.base import Bijection
from normalizing_flows.utils import flatten_event, unflatten_event, get_batch_shape


class Inverse(Bijection):
    def __init__(self, bijection: Bijection):
        super().__init__(bijection.event_shape)
        self.base_bijection = bijection

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_bijection.inverse(x, context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_bijection.forward(z, context)


def invert(bijection_class: Bijection):
    class InverseBijection(Inverse):
        def __init__(self, *args, **kwargs):
            super().__init__(bijection=bijection_class(*args, **kwargs))

    return InverseBijection


class AutoregressiveLayer(Bijection):
    def __init__(self, conditioner: Conditioner, transformer: Transformer, conditioner_transform: ConditionerTransform):
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


class ForwardMaskedAutoregressiveLayer(AutoregressiveLayer):
    def __init__(self, conditioner: Conditioner, transformer: Transformer, conditioner_transform: ConditionerTransform):
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


class InverseMaskedAutoregressiveLayer(AutoregressiveLayer):
    def __init__(self, conditioner: Conditioner, transformer: Transformer, conditioner_transform: ConditionerTransform):
        super().__init__(conditioner, transformer, conditioner_transform)
        self.forward_layer = ForwardMaskedAutoregressiveLayer(
            conditioner,
            transformer,
            conditioner_transform
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_layer.inverse(x, context=context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_layer.forward(z, context=context)


class ElementwiseLayer(AutoregressiveLayer):
    def __init__(self, transformer: Transformer, n_transformer_parameters: int):
        super().__init__(NullConditioner(), transformer, Constant(transformer.event_shape, n_transformer_parameters))
