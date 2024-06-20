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


class Squeeze(Bijection):
    """
    Squeeze a batch of tensors with shape (*batch_shape, channels, height, width) into shape
        (*batch_shape, 4 * channels, height / 2, width / 2).
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        # Check shape length
        if len(event_shape) != 3:
            raise ValueError(f"Event shape must have three components, but got {len(event_shape)}")
        # Check that height and width are divisible by two
        if event_shape[1] % 2 != 0:
            raise ValueError(f"Event dimension 1 must be divisible by 2, but got {event_shape[1]}")
        if event_shape[2] % 2 != 0:
            raise ValueError(f"Event dimension 2 must be divisible by 2, but got {event_shape[2]}")
        super().__init__(event_shape, **kwargs)
        c, h, w = event_shape
        self.squeezed_event_shape = torch.Size((4 * c, h // 2, w // 2))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Squeeze tensor with shape (*batch_shape, channels, height, width) into tensor with shape
            (*batch_shape, 4 * channels, height // 2, width // 2).
        """
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(*batch_shape, device=x.device, dtype=x.dtype)

        channels, height, width = x.shape[-3:]
        assert height % 2 == 0
        assert width % 2 == 0
        n_rows = height // 2
        n_cols = width // 2
        n_squares = n_rows * n_cols

        square_mask = torch.kron(
            torch.arange(n_squares).view(n_rows, n_cols),
            torch.ones(2, 2)
        )
        channel_mask = torch.arange(n_rows * n_cols).view(n_rows, n_cols)[None].repeat(4 * channels, 1, 1)

        # out = torch.zeros(size=(*batch_shape, self.squeezed_event_shape), device=x.device, dtype=x.dtype)
        out = torch.empty(size=(*batch_shape, 4 * channels, height // 2, width // 2), device=x.device, dtype=x.dtype)

        channel_mask = channel_mask.repeat(*batch_shape, 1, 1, 1)
        square_mask = square_mask.repeat(*batch_shape, channels, 1, 1)
        for i in range(n_squares):
            out[channel_mask == i] = x[square_mask == i]

        return out, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Squeeze tensor with shape (*batch_shape, 4 * channels, height // 2, width // 2) into tensor with shape
            (*batch_shape, channels, height, width).
        """
        batch_shape = get_batch_shape(z, self.squeezed_event_shape)
        log_det = torch.zeros(*batch_shape, device=z.device, dtype=z.dtype)

        four_channels, half_height, half_width = z.shape[-3:]
        assert four_channels % 4 == 0
        width = 2 * half_width
        height = 2 * half_height
        channels = four_channels // 4

        n_rows = height // 2
        n_cols = width // 2
        n_squares = n_rows * n_cols

        square_mask = torch.kron(
            torch.arange(n_squares).view(n_rows, n_cols),
            torch.ones(2, 2)
        )
        channel_mask = torch.arange(n_rows * n_cols).view(n_rows, n_cols)[None].repeat(4 * channels, 1, 1)
        out = torch.empty(size=(*batch_shape, channels, height, width), device=z.device, dtype=z.dtype)

        channel_mask = channel_mask.repeat(*batch_shape, 1, 1, 1)
        square_mask = square_mask.repeat(*batch_shape, channels, 1, 1)
        for i in range(n_squares):
            out[square_mask == i] = z[channel_mask == i]

        return out, log_det


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
