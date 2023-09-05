import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import RationalQuadratic
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear_rational import LinearRational
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear import Linear
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.cubic import Cubic
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.basis import Basis


def test_linear_rational():
    torch.manual_seed(0)
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    spline = LinearRational(event_shape=(1,))
    h = torch.randn(size=(len(x), spline.n_parameters))
    z, log_det_forward = spline.forward(x, h)
    xr, log_det_inverse = spline.inverse(z, h)
    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr)
    assert torch.allclose(log_det_forward, -log_det_inverse)


@pytest.mark.parametrize('spline_class', [Linear, LinearRational, RationalQuadratic, Cubic, Basis])
def test_1d_spline(spline_class):
    torch.manual_seed(0)
    spline = spline_class(event_shape=(1,), n_bins=8, boundary=5.0)
    x = torch.tensor([
        [1.2],
        [4.0],
        [-3.6]
    ])
    h = torch.randn(size=(3, 1, spline.n_parameters))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


@pytest.mark.parametrize('spline_class', [RationalQuadratic, LinearRational, Linear, Cubic, Basis])
def test_2d_spline(spline_class):
    torch.manual_seed(0)
    batch_shape = (3,)
    event_shape = (2,)

    spline = spline_class(event_shape=event_shape, n_bins=8, boundary=5.0)
    x = torch.tensor([
        [1.2, 2.0],
        [4.0, 6.0],
        [-3.6, 0.7]
    ])
    h = torch.randn(size=(*batch_shape, *event_shape, spline.n_parameters))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


@pytest.mark.parametrize('boundary', [1.0, 5.0, 50.0])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (10,), (100,), (2, 5, 6, 3)])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (100,), (3, 4, 1)])
@pytest.mark.parametrize('spline_class', [RationalQuadratic, LinearRational, Linear, Cubic, Basis])
def test_spline_exhaustive(spline_class, boundary: float, batch_shape, event_shape):
    torch.manual_seed(0)

    spline = spline_class(event_shape=event_shape, n_bins=8, boundary=boundary)
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, spline.n_parameters))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))
    assert z.shape == x.shape
    assert log_det.shape == batch_shape

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


@pytest.mark.parametrize('n_data', [1, 2, 5, 100, 500])
@pytest.mark.parametrize('n_dim', [1, 2, 5, 100, 500])
@pytest.mark.parametrize('n_bins', [2, 4, 8, 16, 32])
@pytest.mark.parametrize('scale', [1e-2, 1e-1, 1, 1e+1, 1e+2])
def test_rq_spline(n_data, n_dim, n_bins, scale):
    torch.manual_seed(0)

    spline = RationalQuadratic(event_shape=torch.Size((n_dim,)), n_bins=n_bins)
    x = torch.randn(n_data, n_dim) * scale
    h = torch.randn(n_data, n_dim, spline.n_parameters)
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
