import pytest
import torch

from normalizing_flows.src.bijections import AffineCoupling, LinearAffineCoupling, RQSCoupling, LinearRQSCoupling


@pytest.mark.parametrize('layer_class', [AffineCoupling, LinearAffineCoupling])
def test_affine_coupling_too_few_dimensions(layer_class):
    torch.manual_seed(0)
    n_dim = 1
    with pytest.raises(ValueError):
        bijection = layer_class(event_shape=torch.Size((n_dim,)))


@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('layer_class', [AffineCoupling, LinearAffineCoupling])
def test_affine_coupling(n_dim, layer_class):
    torch.manual_seed(0)
    bijection = layer_class(event_shape=torch.Size((n_dim,)))

    x = torch.randn(size=(125, n_dim))
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    assert torch.allclose(log_det_forward, -log_det_inverse), \
        f"{float((log_det_forward+log_det_inverse).abs().max()) = }"
    assert torch.allclose(x, x_reconstructed, atol=1e-5)


@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('layer_class', [RQSCoupling, LinearRQSCoupling])
def test_spline_coupling(n_dim, layer_class):
    torch.manual_seed(0)
    bijection = layer_class(event_shape=torch.Size((n_dim,)))

    x = torch.randn(size=(125, n_dim))
    z, log_det_forward = bijection(x)

    assert x.shape == z.shape
    assert log_det_forward.shape == (x.shape[0],)

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert x_reconstructed.shape == x.shape
    assert log_det_inverse.shape == (x.shape[0],)

    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-4), \
        f"{float((log_det_forward+log_det_inverse).abs().max()) = }"
    assert torch.allclose(x, x_reconstructed, atol=1e-4), f"{float((x-x_reconstructed).abs().max()) = }"
