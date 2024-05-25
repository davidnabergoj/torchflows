from typing import Union, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
from scipy.special import legendre
import tntorch as tn

from normalizing_flows.bijections.numerical_inversion import bisection_no_gradient


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
        assert basis_size > 0
        assert bond_dimension > 0

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=False
        )
        self.event_size = int(torch.prod(torch.as_tensor(event_shape)))
        self.basis_size = basis_size
        self.bond_dimension = bond_dimension
        self.basis = create_orthogonal_polynomials(basis_size)
        self.tt = tn.rand([self.basis_size] * self.event_size, ranks_tt=bond_dimension)

        # Apply left-orthogonalization, end up with QQQQQ...QQQQR
        for core_index in range(self.event_size - 1):
            self.tt.left_orthogonalize(mu=core_index)

        # Check orthonormality
        for core_index in range(self.event_size - 1):
            if not is_orthonormal(self.tt.cores[core_index]):
                raise ValueError(f"Core {core_index} not orthonormal")

    @staticmethod
    def sample_marginal(sample_shape: torch.Size, cdf_1d: callable) -> torch.Tensor:
        # Sample uniform random numbers
        u = torch.rand(size=sample_shape)

        # Apply bisection to the CDF
        return bisection_no_gradient(cdf_1d, u)  # TODO use the gradient to optimize TT cores

    def apply_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        :param inputs: tensor with shape (*b, event_size)
        :return: tensor with shape (*b, event_size, basis_size)
        """
        transformed = torch.zeros(size=(*inputs.shape[:-1], self.event_size, self.basis_size))
        for i in range(self.basis_size):
            transformed[..., i] = self.basis[i](inputs)
        return transformed

    def contract(self, phi: torch.Tensor):
        """

        :param phi: tensor with shape (*b, event_size, basis_size); note: event_size = n_cores.
        :return:
        """
        densities = torch.zeros(size=(self.event_size, *phi.shape[:-2]))

        # Contract with rightmost core (R)
        core_index = self.event_size - 1
        v = torch.einsum('ij,...j->...i', self.tt.cores[core_index][..., 0], phi[..., core_index, :])
        matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
        matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
        f = torch.einsum('...ij,...i,...j->...', matrices_a, phi[..., core_index, :], phi[..., core_index, :])
        densities[core_index] = f

        # Contract with orthonormal cores (Q)
        for core_index in range(len(self.tt.cores) - 2, -1, -1):
            matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
            matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
            f = torch.einsum('...ij,...i,...j->...', matrices_a, phi[..., core_index, :], phi[..., core_index, :])
            densities[core_index] = f
            v = torch.einsum('ijk,...k,...j->...i', self.tt.cores[core_index], v, phi[..., core_index, :])

        # densities[0] is the joint density of all dimensions, i.e. the data density.
        return densities  # The density of input data points

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample dimensions autoregressively with a root-finding algorithm.

        :param sample_shape:
        :return:
        """

        # Sample the last dimension

        # Sample the second-to-last dimension

        # ...

        # Sample the second dimension

        # Sample the first dimension

        pass

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.contract(self.apply_basis(x))[0]


if __name__ == '__main__':
    torch.manual_seed(0)
    base_distribution = TensorTrain(event_shape=(4,), basis_size=3, bond_dimension=7)
    data_points = torch.randn(size=(10, base_distribution.event_size))
    print(base_distribution.log_prob(data_points))
