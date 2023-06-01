import torch

from src.bijections.finite.autoregressive.transformers.spline import RationalQuadraticSpline


def test_invertible():
    torch.manual_seed(0)
    n_bins = 8
    n_data = 5
    n_dim = 2

    x = torch.randn(n_data, n_dim)
    h = torch.randn(n_data, n_dim, 3 * n_bins - 1)
    spline = RationalQuadraticSpline(n_bins=n_bins)
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

    assert torch.allclose(log_det_forward + log_det_inverse, torch.zeros_like(log_det_forward), atol=1e-5)
