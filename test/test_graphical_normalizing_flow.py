import pytest
import torch
from torchflows.architectures import RealNVP, NICE, CouplingRQNSF
from test.constants import __test_constants


@pytest.mark.parametrize('architecture', [RealNVP, NICE, CouplingRQNSF])
def test_basic_2d(architecture):
    torch.manual_seed(0)

    n_data = 100
    n_dim = 2
    x = torch.randn(size=(n_data, n_dim))
    bijection = architecture(event_shape=(n_dim,), edge_list=[(0, 1)])
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert torch.allclose(x, x_reconstructed,
                          atol=__test_constants['data_atol']), f"{torch.linalg.norm(x - x_reconstructed)}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=__test_constants['log_det_atol'])


@pytest.mark.parametrize('architecture', [RealNVP, NICE, CouplingRQNSF])
def test_basic_5d(architecture):
    torch.manual_seed(0)

    n_data = 100
    n_dim = 5
    x = torch.randn(size=(n_data, n_dim))
    bijection = architecture(event_shape=(n_dim,), edge_list=[(0, 1), (0, 2), (0, 3), (0, 4)])
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert torch.allclose(x, x_reconstructed, atol=__test_constants['data_atol'])
    assert torch.allclose(log_det_forward, -log_det_inverse, __test_constants['log_det_atol'])


@pytest.mark.parametrize('architecture', [RealNVP, NICE, CouplingRQNSF])
def test_basic_5d_2(architecture):
    torch.manual_seed(0)

    n_data = 100
    n_dim = 5
    x = torch.randn(size=(n_data, n_dim))
    bijection = architecture(event_shape=(n_dim,), edge_list=[(0, 1)])
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert torch.allclose(x, x_reconstructed, atol=__test_constants['data_atol'])
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=__test_constants['log_det_atol'])


@pytest.mark.parametrize('architecture', [RealNVP, NICE, CouplingRQNSF])
def test_basic_5d_3(architecture):
    torch.manual_seed(0)

    n_data = 100
    n_dim = 5
    x = torch.randn(size=(n_data, n_dim))
    bijection = architecture(event_shape=(n_dim,), edge_list=[(0, 2), (1, 3), (1, 4)])
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert torch.allclose(
        x,
        x_reconstructed,
        atol=__test_constants['data_atol']
    ), f"{torch.linalg.norm(x - x_reconstructed)}"
    assert torch.allclose(
        log_det_forward,
        -log_det_inverse,
        atol=__test_constants['log_det_atol']
    ), f"{torch.linalg.norm(log_det_forward + log_det_inverse)}"


@pytest.mark.parametrize('architecture', [RealNVP, NICE, CouplingRQNSF])
def test_random(architecture):
    torch.manual_seed(0)

    n_data = 100
    n_dim = 50
    x = torch.randn(size=(n_data, n_dim))

    interacting_dimensions = torch.unique(torch.randint(low=0, high=n_dim, size=(n_dim,)))
    interacting_dimensions = interacting_dimensions[torch.randperm(len(interacting_dimensions))]
    source_dimensions = interacting_dimensions[:len(interacting_dimensions) // 2]
    target_dimensions = interacting_dimensions[len(interacting_dimensions) // 2:]

    edge_list = []
    for s in source_dimensions:
        for t in target_dimensions:
            edge_list.append((s, t))

    bijection = architecture(event_shape=(n_dim,), edge_list=edge_list)
    z, log_det_forward = bijection.forward(x)
    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert torch.allclose(
        x,
        x_reconstructed,
        atol=__test_constants['data_atol']
    ), f"{torch.linalg.norm(x - x_reconstructed)}"
    assert torch.allclose(
        log_det_forward,
        -log_det_inverse,
        atol=__test_constants['log_det_atol']
    ), f"{torch.linalg.norm(log_det_forward + log_det_inverse)}"
