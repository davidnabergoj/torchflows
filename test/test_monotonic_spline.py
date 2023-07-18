import pytest
import torch

from normalizing_flows.src.bijections.finite.autoregressive.transformers.spline import MonotonicPiecewiseLinearSpline, \
    MonotonicPiecewiseQuadraticSpline


def test_piecewise_linear_1d():
    torch.manual_seed(0)
    spline = MonotonicPiecewiseLinearSpline(event_shape=(1,), n_bins=8, bound=5.0)
    x = torch.tensor([
        [1.2],
        [4.0],
        [-3.6]
    ])
    h = torch.randn(size=(3, 1, 8))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


def test_piecewise_linear_2d():
    torch.manual_seed(0)
    batch_shape = (3,)
    event_shape = (2,)

    spline = MonotonicPiecewiseLinearSpline(event_shape=event_shape, n_bins=8, bound=5.0)
    x = torch.tensor([
        [1.2, 2.0],
        [4.0, 6.0],
        [-3.6, 0.7]
    ])
    h = torch.randn(size=(*batch_shape, *event_shape, 8))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


@pytest.mark.parametrize('bound', [1.0, 5.0, 50.0])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (10,), (100,), (23, 56, 6, 12)])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (100,), (11, 13, 3)])
def test_piecewise_linear_exhaustive(bound: float, batch_shape, event_shape):
    torch.manual_seed(0)

    spline = MonotonicPiecewiseLinearSpline(event_shape=event_shape, n_bins=8, bound=5.0)
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, spline.n_bins))
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

def test_piecewise_quadratic_1d():
    torch.manual_seed(0)
    batch_shape = (1,)
    event_shape = (1,)

    spline = MonotonicPiecewiseQuadraticSpline(event_shape=event_shape, n_bins=8, bound=5.0)
    x = torch.tensor([[1.2]])
    h = torch.randn(size=(*batch_shape, *event_shape, 3 * 8 + 2))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)

def test_piecewise_quadratic_2d():
    torch.manual_seed(0)
    batch_shape = (3,)
    event_shape = (2,)

    spline = MonotonicPiecewiseQuadraticSpline(event_shape=event_shape, n_bins=8, bound=5.0)
    x = torch.tensor([
        [1.2, 2.0],
        [4.0, 6.0],
        [-3.6, 0.7]
    ])
    h = torch.randn(size=(*batch_shape, *event_shape, 3 * 8 + 2))
    z, log_det = spline(x, h)
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det))

    xr, log_det_inverse = spline.inverse(z, h)
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.allclose(x, xr, atol=1e-3)
    assert torch.allclose(log_det, -log_det_inverse, atol=1e-3)


@pytest.mark.parametrize('bound', [1.0, 5.0, 50.0])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (10,), (100,), (23, 56, 6, 12)])
@pytest.mark.parametrize('event_shape', [(1,), (2,), (10,), (100,), (11, 13, 3)])
def test_piecewise_quadratic_exhaustive(bound: float, batch_shape, event_shape):
    torch.manual_seed(0)

    spline = MonotonicPiecewiseQuadraticSpline(event_shape=event_shape, n_bins=8, bound=5.0)
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, 3 * spline.n_bins + 2))
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
