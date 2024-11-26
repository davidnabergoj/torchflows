from typing import Union, Tuple

import torch

from torchflows.bijections.finite.matrix.base import InvertibleMatrix
from torchflows.bijections.finite.matrix.orthogonal import HouseholderOrthogonalMatrix
from torchflows.bijections.finite.matrix.triangular import LowerTriangularInvertibleMatrix, \
    UpperTriangularInvertibleMatrix


class LUMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.lower = LowerTriangularInvertibleMatrix(self.n_dim, unitriangular=True, **kwargs)
        self.upper = UpperTriangularInvertibleMatrix(self.n_dim, **kwargs)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return self.lower.project_flat(self.upper.project_flat(x_flat))

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.upper.solve_flat(self.lower.solve_flat(b_flat))

    def log_det_project(self):
        return self.upper.log_det_project()


class QRMatrix(InvertibleMatrix):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]], **kwargs):
        super().__init__(event_shape, **kwargs)
        self.orthogonal = HouseholderOrthogonalMatrix(self.n_dim, **kwargs)
        self.upper = UpperTriangularInvertibleMatrix(self.n_dim, **kwargs)

    def project_flat(self, x_flat: torch.Tensor, context_flat: torch.Tensor = None) -> torch.Tensor:
        return self.orthogonal.project_flat(self.upper.project_flat(x_flat))

    def solve_flat(self, b_flat: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        w_flat = self.orthogonal.solve_flat(b_flat)
        x_flat = self.upper.solve_flat(w_flat)
        return x_flat

    def log_det_project(self):
        return self.upper.log_det_project()
