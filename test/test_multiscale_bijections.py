from torchflows.architectures import (
    MultiscaleNICE,
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleRealNVP
)
import torch
import pytest
from test.constants import __test_constants


@pytest.mark.parametrize('architecture_class', [
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    MultiscaleRealNVP
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_basic(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape, n_layers=2)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert torch.allclose(x, xr, atol=__test_constants['data_atol'])
    assert torch.allclose(ldf, -ldi, atol=__test_constants['log_det_atol'])  # 1e-2


@pytest.mark.parametrize('architecture_class', [
    MultiscaleRQNSF,
    MultiscaleLRSNSF,
    MultiscaleNICE,
    MultiscaleRealNVP
])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_too_small_image(architecture_class, image_shape):
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        bijection = architecture_class(image_shape, n_layers=3)









