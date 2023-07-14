import torch

from normalizing_flows.src.bijections.finite.autoregressive.transformers.spline import MonotonicPiecewiseLinearSpline


def test_1d():
    spline = MonotonicPiecewiseLinearSpline(event_shape=(1,), n_bins=8, bin_width=1)
    x = torch.tensor([
        [1.2],
        [4.0],
        [-3.6]
    ])
    h = torch.randn(size=(3, 1, 8))
    z, log_det = spline(x, h)


def test_2d():
    batch_shape = (3,)
    event_shape = (2,)

    spline = MonotonicPiecewiseLinearSpline(event_shape=event_shape, n_bins=8, bin_width=1)
    x = torch.tensor([
        [1.2, 2.0],
        [4.0, 6.0],
        [-3.6, 0.7]
    ])
    h = torch.randn(size=(*batch_shape, *event_shape, 8))
    z, log_det = spline(x, h)
