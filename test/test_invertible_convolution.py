import torch
from torchflows.bijections.finite.multiscale.base import Invertible1x1ConvolutionalCoupling

def test_basic():
    torch.manual_seed(0)
    event_shape = 3, 20, 20
    x = torch.randn(size=(4, *event_shape))
    layer = Invertible1x1ConvolutionalCoupling(event_shape)
    z, log_det = layer.forward(x)
    xr, log_det_inv = layer.inverse(z)

    assert torch.allclose(x, xr)
    assert torch.allclose(log_det_inv, -log_det)