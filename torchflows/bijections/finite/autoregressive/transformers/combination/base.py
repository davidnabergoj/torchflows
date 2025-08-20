import torch
from typing import Tuple, List, Union

from torchflows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from torchflows.utils import get_batch_shape


class Combination(ScalarTransformer):
    def __init__(self, event_shape: torch.Size, components: List[ScalarTransformer]):
        super().__init__(event_shape)
        self.components = components
        self.n_components = len(self.components)

    @property
    def parameter_shape_per_element(self) -> Union[torch.Size, Tuple[int, ...]]:
        return (sum([c.n_parameters_per_element for c in self.components]),)

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.cat([c.default_parameters.ravel() for c in self.components], dim=0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(size=batch_shape).to(x)
        start_index = 0
        for i in range(self.n_components):
            component = self.components[i]
            x, log_det_increment = component.forward(x, h[..., start_index:start_index + component.n_parameters_per_element])
            log_det += log_det_increment
            start_index += component.n_parameters_per_element
        z = x
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(size=batch_shape).to(z)
        c = self.n_parameters_per_element
        for i in range(self.n_components):
            component = self.components[self.n_components - i - 1]
            c -= component.n_parameters_per_element
            z, log_det_increment = component.inverse(z, h[..., c:c + component.n_parameters_per_element])
            log_det += log_det_increment
        x = z
        return x, log_det
