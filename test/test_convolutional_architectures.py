from torchflows.bijections.finite.multiscale import (
    MultiscaleNICE,
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleRealNVP,
    AffineGlow,
    RQSGlow,
    LRSGlow,
    ShiftGlow,
)
from torchflows.bijections.continuous import (
    ConvolutionalRNODE,
    ConvolutionalDDNF,
    ConvolutionalFFJORD,
)

import torch
import pytest
from test.constants import __test_constants
from torchflows.bijections.finite.residual.architectures import ConvolutionalInvertibleResNet, ConvolutionalResFlow


@pytest.mark.parametrize('architecture_class', [
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    MultiscaleRealNVP,
    AffineGlow,
    RQSGlow,
    LRSGlow,
    ShiftGlow
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_autoregressive(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape, n_layers=2)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert torch.allclose(x, xr, atol=__test_constants['data_atol'])
    assert torch.allclose(ldf, -ldi, atol=__test_constants['log_det_atol'])  # 1e-2


@pytest.mark.skip('Unsupported/failing')
@pytest.mark.parametrize('architecture_class', [
    ConvolutionalRNODE,
    ConvolutionalDDNF,
    ConvolutionalFFJORD
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_continuous(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert x.shape == xr.shape
    assert ldf.shape == ldi.shape
    assert torch.allclose(x, xr, atol=__test_constants['data_atol']), f'"{(x - xr).abs().max()}"'
    assert torch.allclose(ldf, -ldi, atol=__test_constants['log_det_atol'])  # 1e-2


@pytest.mark.skip('Unsupported/failing')
@pytest.mark.parametrize('architecture_class', [
    ConvolutionalInvertibleResNet,
    ConvolutionalResFlow
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_residual(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert x.shape == xr.shape
    assert ldf.shape == ldi.shape
    assert torch.allclose(x, xr, atol=__test_constants['data_atol']), f'"{(x - xr).abs().max()}"'
    assert torch.allclose(ldf, -ldi, atol=__test_constants['log_det_atol'])  # 1e-2


@pytest.mark.parametrize('architecture_class', [
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    MultiscaleRealNVP,
    AffineGlow,
    RQSGlow,
    LRSGlow,
    ShiftGlow
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_too_small_image(architecture_class, image_shape):
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        bijection = architecture_class(image_shape, n_layers=3)
