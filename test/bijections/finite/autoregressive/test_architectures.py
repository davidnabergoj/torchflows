import pytest
import torch

from src.bijections.finite.autoregressive.architectures import RealNVP, MAF


@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('architecture_class', [RealNVP, MAF])
def test_architecture(n_dim, architecture_class):
    torch.manual_seed(0)
    bijection = architecture_class(n_dim)

    x = torch.randn(size=(125, n_dim))
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-6)
    assert torch.allclose(x, x_reconstructed, atol=1e-2)
