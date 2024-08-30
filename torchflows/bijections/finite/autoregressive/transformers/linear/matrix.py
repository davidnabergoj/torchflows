from typing import Union, Tuple

import torch

from torchflows.bijections.finite.autoregressive.transformers.base import TensorTransformer
from torchflows.utils import flatten_event, unflatten_event


# Matrix transformers that operate on vector inputs (Ax=b)

class LUTransformer(TensorTransformer):
    """Linear transformer with LUx = y.

    It is assumed that all diagonal elements of L are 1.
    """

    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    def extract_matrices(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract matrices L, U from tensor h.

        :param torch.Tensor h: parameter tensor with shape (*batch_shape, *parameter_shape)
        :returns: tuple with (L, U, log(diag(U))). L and U have shapes (*batch_shape, event_size, event_size),
         log(diag(U)) has shape (*batch_shape, event_size).
        """
        event_size = int(torch.prod(torch.as_tensor(self.event_shape)))
        n_off_diag_el = (event_size ** 2 - event_size) // 2

        u_unc_diag = h[..., :event_size]
        u_diag = torch.exp(u_unc_diag) / 10 + 1
        u_log_diag = torch.log(u_diag)

        u_off_diagonal_elements = h[..., event_size:event_size + n_off_diag_el] / 10
        l_off_diagonal_elements = h[..., -n_off_diag_el:] / 10

        batch_shape = h.shape[:-len(self.parameter_shape)]

        upper = torch.zeros(size=(*batch_shape, event_size, event_size)).to(h)
        upper_row_index, upper_col_index = torch.triu_indices(row=event_size, col=event_size, offset=1)
        upper[..., upper_row_index, upper_col_index] = u_off_diagonal_elements
        upper[..., range(event_size), range(event_size)] = u_diag

        lower = torch.zeros(size=(*batch_shape, event_size, event_size)).to(h)
        lower_row_index, lower_col_index = torch.tril_indices(row=event_size, col=event_size, offset=-1)
        lower[..., lower_row_index, lower_col_index] = l_off_diagonal_elements
        lower[..., range(event_size), range(event_size)] = 1  # Unit diagonal

        return lower, upper, u_log_diag

    @staticmethod
    def log_determinant(upper_log_diag: torch.Tensor):
        """
        Computes the matrix log determinant of A = LU for each pair of matrices in a batch.

        Note: det(A) = det(LU) = det(L) * det(U) so log det(A) = log det(L) + log det(U).
        We assume that L has unit diagonal, so log det(L) = 0 and can be skipped.

        :param torch.Tensor upper_log_diag: log diagonals of matrices U with shape (*batch_size, event_size).
        :returns: log determinants of LU with shape (*batch_size,).
        """
        # Extract the diagonals
        return torch.sum(upper_log_diag, dim=-1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper, upper_log_diag = self.extract_matrices(h)

        # Flatten inputs
        x_flat = flatten_event(x, self.event_shape)  # (*batch_shape, event_size)
        y_flat = torch.einsum('...ij,...jk,...k->...i', lower, upper, x_flat)  # y = LUx

        output = unflatten_event(y_flat, self.event_shape)
        return output, self.log_determinant(upper_log_diag)

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper, upper_log_diag = self.extract_matrices(h)

        # Flatten inputs
        y_flat = flatten_event(y, self.event_shape)[..., None]  # (*batch_shape, event_size)
        z_flat = torch.linalg.solve_triangular(lower, y_flat, upper=False, unitriangular=True)  # y = Lz => z = L^{-1}y
        x_flat = torch.linalg.solve_triangular(upper, z_flat, upper=True, unitriangular=False)  # z = Ux => x = U^{-1}z
        x_flat = x_flat.squeeze(-1)

        output = unflatten_event(x_flat, self.event_shape)
        return output, -self.log_determinant(upper_log_diag)

    @property
    def parameter_shape(self) -> Union[torch.Size, Tuple[int, ...]]:
        event_size = int(torch.prod(torch.as_tensor(self.event_shape)))
        # Let n be the event size
        # L will have (n^2 - n) / 2 parameters (we assume unit diagonal)
        # U will have (n^2 - n) / 2 + n parameters
        n_off_diag_el = (event_size ** 2 - event_size) // 2
        return (event_size + n_off_diag_el + n_off_diag_el,)

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=self.parameter_shape)
