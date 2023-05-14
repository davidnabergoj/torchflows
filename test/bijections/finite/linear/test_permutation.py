import pytest
import torch

from src.bijections.finite.linear.permutation import Permutation


@pytest.mark.parametrize('n_dim', [1, 2, 10, 100])
def test_permutation(n_dim):
    torch.manual_seed(0)
    x = torch.randn(25, n_dim)
    bijection = Permutation(n_dim=n_dim)

    z, log_det_forward = bijection(x)

    assert log_det_forward.shape == (x.shape[0],)
    assert torch.allclose(log_det_forward, torch.zeros_like(log_det_forward))

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert log_det_inverse.shape == (x.shape[0],)
    assert torch.allclose(log_det_inverse, torch.zeros_like(log_det_inverse))

    assert torch.allclose(x, x_reconstructed)
    assert torch.allclose(torch.unique(x), torch.unique(z))
