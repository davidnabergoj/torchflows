from normalizing_flows.architectures import MultiscaleNICE, MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleRealNVP
import torch
import pytest


@pytest.mark.parametrize('architecture_class', [MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleNICE, MultiscaleRealNVP])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_non_factored(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape, n_layers=2, factored=False)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert torch.allclose(x, xr, atol=1e-4)
    assert torch.allclose(ldf, -ldi, atol=1e-2)


@pytest.mark.parametrize('architecture_class', [MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleNICE, MultiscaleRealNVP])
@pytest.mark.parametrize('image_shape', [(1, 28, 28), (3, 28, 28)])
def test_non_factored_too_small_image(architecture_class, image_shape):
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        bijection = architecture_class(image_shape, n_layers=3, factored=False)


@pytest.mark.parametrize('architecture_class', [MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleNICE, MultiscaleRealNVP])
@pytest.mark.parametrize('image_shape', [(1, 32, 32), (3, 32, 32)])
def test_factored(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    bijection = architecture_class(image_shape, n_layers=2, factored=True)
    z, ldf = bijection.forward(x)
    xr, ldi = bijection.inverse(z)
    assert torch.allclose(x, xr, atol=1e-4)
    assert torch.allclose(ldf, -ldi, atol=1e-2)


@pytest.mark.parametrize('architecture_class', [MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleNICE, MultiscaleRealNVP])
@pytest.mark.parametrize('image_shape', [(1, 15, 32), (3, 15, 32)])
def test_factored_wrong_shape(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    with pytest.raises(ValueError):
        bijection = architecture_class(image_shape, n_layers=2, factored=True)


@pytest.mark.parametrize('architecture_class', [MultiscaleRQNSF, MultiscaleLRSNSF, MultiscaleNICE, MultiscaleRealNVP])
@pytest.mark.parametrize('image_shape', [(1, 8, 8), (3, 8, 8)])
def test_factored_too_small_image(architecture_class, image_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(5, *image_shape))
    with pytest.raises(ValueError):
        bijection = architecture_class(image_shape, n_layers=8, factored=True)
