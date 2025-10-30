import torch

from torchflows.bijections.finite.residual import RadialBijection


def test_exhaustive():
    torch.manual_seed(0)
    event_shape = (1000,)
    bijection = RadialBijection(event_shape=event_shape)
    z = torch.randn(size=(5000, *event_shape)) ** 2
    x, log_det_inverse = bijection.inverse(z)
    assert torch.isfinite(x).all()
    assert torch.isfinite(log_det_inverse).all()
