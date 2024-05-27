from typing import Union, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
from scipy.special import legendre
import tntorch as tn

from normalizing_flows.bijections.numerical_inversion import bisection_no_gradient
from normalizing_flows.utils import sum_except_batch


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
        transformed = torch.zeros(size=(*inputs.shape, self.basis_size))
        for i in range(self.basis_size):
            transformed[..., i] = self.basis[i](inputs)
        return transformed

    def contract(self, phi: torch.Tensor):
        """

        :param phi: tensor with shape (*b, event_size, basis_size); note: event_size = n_cores.
        :return:
        """
        fs = torch.zeros(size=(self.event_size, *phi.shape[:-2]))

        # Contract with rightmost core (R)
        core_index = self.event_size - 1
        v = torch.einsum('ij,...j->...i', self.tt.cores[core_index][..., 0], phi[..., core_index, :])
        matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
        matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
        f = torch.einsum('...ij,...i,...j->...', matrices_a, phi[..., core_index, :], phi[..., core_index, :])
        fs[core_index] = f

        # Contract with orthonormal cores (Q)
        for core_index in range(len(self.tt.cores) - 2, -1, -1):
            matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
            matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
            f = torch.einsum('...ij,...i,...j->...', matrices_a, phi[..., core_index, :], phi[..., core_index, :])
            fs[core_index] = f
            v = torch.einsum('ijk,...k,...j->...i', self.tt.cores[core_index], v, phi[..., core_index, :])

        # fs[0] is the joint density of all dimensions, i.e. the data density.
        return fs  # The density of input data points

    def compute_f_v(self, x: torch.Tensor, v: torch.Tensor, core_index: int):
        phi = self.apply_basis(x)
        matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
        matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
        f = torch.einsum('...ij,...i,...j->...', matrices_a, phi, phi)
        v = torch.einsum('ijk,...k,...j->...i', self.tt.cores[core_index], v, phi)
        return f, v

    def compute_f_v_first(self, x: torch.Tensor):
        core_index = self.event_size - 1
        phi = self.apply_basis(x)
        v = torch.einsum('ij,...j->...i', self.tt.cores[core_index][..., 0], phi)
        matrices_b = torch.einsum('...k,ijk->...ji', v, self.tt.cores[core_index])
        matrices_a = torch.einsum('...ij,...kj->...ik', matrices_b, matrices_b)
        f = torch.einsum('...ij,...i,...j->...', matrices_a, phi, phi)
        return f, v

    def sample_dim(self,
                   dim: int,
                   sample_shape: torch.Size,
                   v: torch.Tensor = None,
                   n_grid_points: int = 1000):
        x_grid = torch.linspace(-1, 1, steps=n_grid_points)
        if dim == self.event_size - 1:
            f, v = self.compute_f_v_first(x_grid)
        else:
            f, v = self.compute_f_v(x_grid, v, core_index=dim)
        cdf_scaled = torch.cumsum(f, dim=0)
        cdf_min = torch.min(cdf_scaled)
        cdf_max = torch.max(cdf_scaled)
        cdf = (cdf_scaled - cdf_min) / (cdf_max - cdf_min)
        u = torch.rand(size=sample_shape)
        x_indices = torch.argmin(torch.as_tensor(u[..., None] >= cdf[[None] * len(sample_shape)]).to(u), dim=-1)
        x_dim = x_grid[x_indices]
        v_dim = v
        f_dim = f[x_indices]
        return x_dim, v_dim, f_dim

    def sample_with_log_prob(self, sample_shape: Union[torch.Size, Tuple[int, ...]] = torch.Size()) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Sample dimensions autoregressively with a root-finding algorithm.

        :param sample_shape:
        :return:
        """
        x = torch.zeros(size=(*sample_shape, self.event_size))

        # Contract with rightmost core (R)
        core_index = self.event_size - 1
        x_d, v_d, f_d = self.sample_dim(core_index, sample_shape=sample_shape)
        x[..., core_index] = x_d

        # Contract with orthonormal cores (Q)
        for core_index in range(len(self.tt.cores) - 2, -1, -1):
            # f, v = self.compute_f_v(x[..., core_index], v, core_index)
            x_d, v_d, f_d = self.sample_dim(core_index, sample_shape=sample_shape, v=v_d)
            x[..., core_index] = x_d

        return x, torch.log(f_d)

    def sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]] = torch.Size()) -> torch.Tensor:
        """
        Sample dimensions autoregressively with a root-finding algorithm.

        :param sample_shape:
        :return:
        """

        return self.sample_with_log_prob(sample_shape)[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.contract(self.apply_basis(x))[0]


class UnconstrainedTensorTrain(TensorTrain):
    """
    TensorTrain-Based distribution with support for the entire Euclidean space, not only [-1, 1]^d.
    This is done with the inverse TanH transform.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], torch.Size], basis_size: int, bond_dimension: int):
        super().__init__(event_shape, basis_size, bond_dimension)

    @staticmethod
    def inverse_tanh(x: torch.Tensor) -> torch.Tensor:
        x_unconstrained = 0.5 * (torch.log1p(x) - torch.log1p(-x))
        return x_unconstrained

    @staticmethod
    def log_d_dx_tanh(x_unconstrained: torch.Tensor):
        return torch.log1p(-torch.tanh(x_unconstrained) ** 2)

    @staticmethod
    def log_d_dx_tanh_inverse(x_constrained: torch.Tensor):
        return -torch.log1p(-x_constrained ** 2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        base_log_prob = super().log_prob(torch.tanh(x))
        log_det = sum_except_batch(self.log_d_dx_tanh(x), (self.event_size,))
        return base_log_prob + log_det

    def sample_with_log_prob(self,
                             sample_shape: Union[torch.Size, Tuple[int, ...]] = torch.Size()) -> Tuple[
        torch.Tensor, torch.Tensor]:
        x_constrained, base_log_prob = super().sample_with_log_prob(sample_shape)
        x_unconstrained = self.inverse_tanh(x_constrained)
        log_prob = base_log_prob + sum_except_batch(self.log_d_dx_tanh_inverse(x_constrained), (self.event_size,))
        return x_unconstrained, log_prob


if __name__ == '__main__':
    torch.manual_seed(0)
    base_distribution = TensorTrain(event_shape=(4,), basis_size=3, bond_dimension=7)

    # Inputs need to be between -1 and 1 for the Legendre polynomials
    data_points = torch.rand(size=(10, base_distribution.event_size)) * 2 - 1
    print(base_distribution.log_prob(data_points))

    base_samples = base_distribution.sample((10,))
    print(f'{base_samples.shape = }')
    print(torch.isfinite(base_samples).all())
    print(base_samples)

    base_samples, log_prob = base_distribution.sample_with_log_prob((10,))
    print(base_samples.shape)
    print(log_prob.shape)

    base_distribution = UnconstrainedTensorTrain(event_shape=(4,), basis_size=3, bond_dimension=7)
    ret = base_distribution.sample((10,))
    print(ret.shape)
    print(ret)
