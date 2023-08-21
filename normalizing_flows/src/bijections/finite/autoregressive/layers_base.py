from typing import Tuple

import torch

from normalizing_flows.src.bijections.finite.autoregressive.conditioners.base import Conditioner, NullConditioner
from normalizing_flows.src.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform, Constant
from normalizing_flows.src.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.src.bijections.finite.base import Bijection


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
        log_det = torch.zeros(size=(z.shape[0],), device=z.device)
        x = torch.clone(z)
        for i in torch.arange(z.shape[-1]):
            x_clone = torch.clone(x)
            h = self.conditioner(x_clone, transform=self.conditioner_transform, context=context)
            tmp, log_det = self.transformer.inverse(x_clone, h)
            x[:, i] = tmp[:, i]
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
        super().__init__(NullConditioner(), transformer, Constant(n_transformer_parameters))
