import pytest
import torch
from normalizing_flows import InvertibleResNet


@pytest.mark.parametrize('n_iterations', [1, 10, 50, 150])
def test_invertible_resnet(n_iterations: int):
    # We get a good inverse with 150 iterations

    torch.manual_seed(0)

    x = torch.randn(1, 2)
    bijection = InvertibleResNet(event_shape=(2,), n_iterations=n_iterations)
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    print(f'{x = }')
    print(f'{z = }')
    print(f'{log_det_forward = }')

    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(x_reconstructed))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.all(torch.isfinite(z))
    assert torch.all(torch.isfinite(x_reconstructed))
    assert torch.all(torch.isfinite(log_det_forward))
    assert torch.all(torch.isfinite(log_det_inverse))

    assert torch.allclose(x_reconstructed, x)
    assert torch.allclose(log_det_forward, -log_det_inverse)
