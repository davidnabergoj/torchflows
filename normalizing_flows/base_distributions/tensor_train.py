from typing import Union, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
from scipy.special import legendre
import tntorch as tn


def is_orthonormal(core):
    # Core has shape (c1, c2, c3)
    # If core C is orthonormal, then C[..., i] @ C[..., i] = 1 and C[..., i] @ C[..., j] = 0 for i != j.
    product = torch.einsum('iam,ian->mn', core, core)
    return torch.allclose(product, torch.eye(len(product)), atol=1e-4)


class LegendrePolynomial(nn.Module):
    """
    Legendre polynomial. Maps a scalar in [-1, 1] to a scalar in [-1, 1]. Works on batched inputs.
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        assert degree > 0
        self.poly = legendre(degree)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.poly(x.detach().numpy()))


def create_orthogonal_polynomials(n_polynomials):
    # Creates <n_polynomials> Legendre polynomials with degrees [1, 2, ..., <n_polynomials>]
    return [LegendrePolynomial(deg) for deg in range(1, n_polynomials + 1)]


class TensorTrain(dist.Distribution):
    """
    Class that defines a tensor train distribution.

    Reference paper: Khoo et al. "Tensorizing flows: a tool for variational inference" (2023), arxiv: 2305.02460.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], basis_size: int, bond_dimension: int):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=False
        )
        self.event_size = int(torch.prod(torch.as_tensor(event_shape)))

        assert basis_size > 0
        self.basis_size = basis_size

        assert bond_dimension > 0
        self.bond_dimension = bond_dimension

        self.basis = create_orthogonal_polynomials(basis_size)

        self.tt = tn.rand([self.basis_size] * self.event_size, ranks_tt=bond_dimension)

        # Apply left-orthogonalization, end up with QQQQQ...QQQQR

        for core_index in range(self.event_size - 1):
            self.tt.left_orthogonalize(mu=core_index)

        # Check orthonormality
        for core_index in range(self.event_size - 1):
            print(is_orthonormal(self.tt.cores[core_index]))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == '__main__':
    base = TensorTrain(event_shape=(4,), basis_size=3, bond_dimension=7)
