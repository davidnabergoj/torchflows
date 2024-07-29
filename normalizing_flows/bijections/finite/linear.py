import torch

from typing import Tuple, Union

from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.matrices import (
    LowerTriangularInvertibleMatrix,
    HouseholderOrthogonalMatrix,
    InvertibleMatrix,
    PermutationMatrix,
    LUMatrix,
    QRMatrix
)
from normalizing_flows.utils import get_batch_shape, flatten_event, unflatten_event, flatten_batch, unflatten_batch


class Identity(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)
        return x, torch.zeros(size=batch_shape, device=x.device)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)
        return z, torch.zeros(size=batch_shape, device=z.device)


class LinearBijection(Bijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], matrix: InvertibleMatrix):
        super().__init__(event_shape)
        self.matrix = matrix

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(x, self.event_shape)

        x = flatten_batch(flatten_event(x, self.event_shape), batch_shape)  # (n_batch_dims, n_event_dims)
        z = self.matrix.project(x)
        z = unflatten_batch(unflatten_event(z, self.event_shape), batch_shape)

        log_det = self.matrix.log_det() + torch.zeros(size=batch_shape, device=x.device)
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = get_batch_shape(z, self.event_shape)

        z = flatten_batch(flatten_event(z, self.event_shape), batch_shape)  # (n_batch_dims, n_event_dims)
        x = self.matrix.solve(z)
        x = unflatten_batch(unflatten_event(x, self.event_shape), batch_shape)

        log_det = -self.matrix.log_det() + torch.zeros(size=batch_shape, device=z.device)
        return x, log_det


class Permutation(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape, PermutationMatrix(int(torch.prod(torch.as_tensor(event_shape)))))


class ReversePermutation(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        matrix = PermutationMatrix(int(torch.prod(torch.as_tensor(event_shape))))
        matrix.forward_permutation = (matrix.n_dim - 1) - torch.arange(matrix.n_dim)
        matrix.inverse_permutation = torch.empty_like(matrix.forward_permutation)
        matrix.inverse_permutation[matrix.forward_permutation] = torch.arange(matrix.n_dim)
        super().__init__(event_shape, matrix)


class LowerTriangular(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape, LowerTriangularInvertibleMatrix(int(torch.prod(torch.as_tensor(event_shape)))))


class LU(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape, LUMatrix(int(torch.prod(torch.as_tensor(event_shape)))))


class QR(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape, QRMatrix(int(torch.prod(torch.as_tensor(event_shape)))))


class Orthogonal(LinearBijection):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape, HouseholderOrthogonalMatrix(int(torch.prod(torch.as_tensor(event_shape)))))
