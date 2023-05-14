import pytest
import torch

from src.bijections.finite.autoregressive.layers import FeedForwardAffineCoupling


@pytest.mark.parametrize('n_dim', [2, 10, 100])
def test_affine_coupling(n_dim):
    torch.manual_seed(0)
    bijection = FeedForwardAffineCoupling(n_dim)

    x = torch.randn(size=(25, n_dim))
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    assert torch.allclose(log_det_forward, -log_det_inverse)
    assert torch.allclose(x, x_reconstructed, atol=1e-2)
