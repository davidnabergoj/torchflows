import pytest
import torch

from src.bijections.finite.autoregressive.transformers.spline import RationalQuadraticSpline


@pytest.mark.parametrize('n_data', [1, 2, 5, 100, 500])
@pytest.mark.parametrize('n_dim', [1, 2, 5, 100, 500])
@pytest.mark.parametrize('n_bins', [2, 4, 8, 16, 32])
@pytest.mark.parametrize('scale', [1e-2, 1e-1, 1, 1e+1, 1e+2])
def test_invertible(n_data, n_dim, n_bins, scale):
    torch.manual_seed(0)

    x = torch.randn(n_data, n_dim) * scale
    h = torch.randn(n_data, n_dim, 3 * n_bins - 1)
    spline = RationalQuadraticSpline(event_shape=torch.Size((n_dim,)), n_bins=n_bins)
    z, log_det_forward = spline.forward(x, h)

    assert z.shape == x.shape
    assert log_det_forward.shape == (z.shape[0],)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det_forward))

    y, log_det_inverse = spline.inverse(z, h)
    assert y.shape == z.shape
    assert log_det_inverse.shape == log_det_forward.shape
    assert torch.all(~torch.isnan(y))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(log_det_forward + log_det_inverse, torch.zeros_like(log_det_forward), atol=1e-3)
