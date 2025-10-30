import torch

from torchflows.bijections.finite.residual import RadialBijection


def test_exhaustive():
    torch.manual_seed(0)
    event_shape = (1000,)
    bijection = RadialBijection(event_shape=event_shape)
    z = torch.rand(size=(5000, *event_shape)) * 20 - 10  # U(-10, 10)
    x, log_det_inverse = bijection.inverse(z)

    assert x.isfinite().all()
    assert log_det_inverse.isfinite().all()
