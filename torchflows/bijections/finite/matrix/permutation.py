from typing import Union, Tuple

import torch

from torchflows.bijections.finite.matrix.base import InvertibleMatrix


class PermutationMatrix(InvertibleMatrix):
    def __init__(self,
                 event_shape: Union[torch.Size, Tuple[int, ...]],
                 forward_permutation: torch.Tensor,
                 **kwargs):
        super().__init__(event_shape, **kwargs)
        assert forward_permutation.shape == event_shape
        self.forward_permutation = forward_permutation.view(-1)
        self.inverse_permutation = torch.empty_like(self.forward_permutation)
        self.inverse_permutation[self.forward_permutation] = torch.arange(self.n_dim)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return x_flat[..., self.forward_permutation]

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return b_flat[..., self.inverse_permutation]

    def log_det_project(self) -> torch.Tensor:
        return torch.zeros(1).to(self.device_buffer.device)

class RandomPermutationMatrix(PermutationMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        super().__init__(event_shape, forward_permutation=torch.randperm(n_dim).view(*event_shape), **kwargs)


class ReversePermutationMatrix(PermutationMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        super().__init__(event_shape, forward_permutation=torch.arange(n_dim - 1, -1, -1).view(*event_shape), **kwargs)
