from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network import \
    UnconstrainedMonotonicNeuralNetwork
from test.constants import __test_constants


@pytest.mark.skip(reason="Not finalized")
@pytest.mark.parametrize('batch_shape', __test_constants["batch_shape"])
@pytest.mark.parametrize('event_shape', __test_constants["event_shape"])
def test_umnn(batch_shape: Tuple, event_shape: Tuple):
    # Event shape cannot be too big, otherwise
    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape) / 100

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(size=(*batch_shape, *transformer.parameter_shape))
    z, log_det_forward = transformer.forward(x, h)
    xr, log_det_inverse = transformer.inverse(z, h)

    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == batch_shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr, atol=1e-3), f"{torch.max(torch.abs(x-xr)) = }"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3), \
        f"{torch.max(torch.abs(log_det_forward+log_det_inverse)) = }"


@pytest.mark.skip(reason="Not finalized")
def test_umnn_forward():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_forward0 = transformer.forward(x0, h)
    z1, log_det_forward1 = transformer.forward(x1, h)
    z, log_det_forward = transformer.forward(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1]), log_det_forward)


@pytest.mark.skip(reason="Not finalized")
def test_umnn_inverse():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_inverse0 = transformer.inverse(x0, h)
    z1, log_det_inverse1 = transformer.inverse(x1, h)
    z, log_det_inverse = transformer.inverse(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_inverse0, log_det_inverse1]), log_det_inverse)


@pytest.mark.skip(reason="Not finalized")
def test_umnn_reconstruction():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_forward0 = transformer.forward(x0, h)
    z1, log_det_forward1 = transformer.forward(x1, h)
    z, log_det_forward = transformer.forward(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1]), log_det_forward)

    x0r, log_det_inverse0 = transformer.inverse(z0, h)
    x1r, log_det_inverse1 = transformer.inverse(z1, h)
    xr, log_det_inverse = transformer.inverse(z, h.repeat(2, 1))

    assert torch.allclose(x0r, x0, atol=1e-4)
    assert torch.allclose(x1r, x1, atol=1e-4)
    assert torch.allclose(xr, x, atol=1e-4)

    assert torch.allclose(log_det_forward0, -log_det_inverse0, atol=1e-4)
    assert torch.allclose(log_det_forward1, -log_det_inverse1, atol=1e-4)
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-4)


@pytest.mark.skip(reason="Not finalized")
def test_umnn_forward_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2)
    x1 = torch.tensor([1.0, 2.0]).view(1, 2)
    x2 = torch.tensor([5.0, 6.0]).view(1, 2)
    x = torch.cat([x0, x1, x2]).view(3, 2)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_forward0 = transformer.forward(x0, h)
    z1, log_det_forward1 = transformer.forward(x1, h)
    z2, log_det_forward2 = transformer.forward(x2, h)
    z, log_det_forward = transformer.forward(x, h.repeat(3, 1))

    assert torch.allclose(torch.cat([z0, z1, z2]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1, log_det_forward2]), log_det_forward)


@pytest.mark.skip(reason="Not finalized")
def test_umnn_inverse_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2)
    x1 = torch.tensor([1.0, 2.0]).view(1, 2)
    x2 = torch.tensor([5.0, 6.0]).view(1, 2)
    x = torch.cat([x0, x1, x2]).view(3, 2)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_inverse0 = transformer.inverse(x0, h)
    z1, log_det_inverse1 = transformer.inverse(x1, h)
    z2, log_det_inverse2 = transformer.inverse(x2, h)
    z, log_det_inverse = transformer.inverse(x, h.repeat(3, 1))

    assert torch.allclose(torch.cat([z0, z1, z2]), z)
    assert torch.allclose(torch.as_tensor([log_det_inverse0, log_det_inverse1, log_det_inverse2]), log_det_inverse)


@pytest.mark.skip(reason="Not finalized")
def test_umnn_reconstruction_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2) / 10
    x1 = torch.tensor([1.0, 2.0]).view(1, 2) / 10
    x2 = torch.tensor([5.0, 6.0]).view(1, 2) / 10
    x = torch.cat([x0, x1, x2]).view(3, 2)

    transformer = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    h = torch.randn(*event_shape, *transformer.parameter_shape_per_element)

    z0, log_det_forward0 = transformer.forward(x0, h)
    z1, log_det_forward1 = transformer.forward(x1, h)
    z2, log_det_forward2 = transformer.forward(x2, h)
    z, log_det_forward = transformer.forward(x, h.repeat(3, 1))

    x0r, log_det_inverse0 = transformer.inverse(z0, h)
    x1r, log_det_inverse1 = transformer.inverse(z1, h)
    x2r, log_det_inverse2 = transformer.inverse(z2, h)
    xr, log_det_inverse = transformer.inverse(z, h.repeat(3, 1))

    assert torch.allclose(x0r, x0, atol=1e-3)
    assert torch.allclose(x1r, x1, atol=1e-3)
    assert torch.allclose(x2r, x2, atol=1e-3)
    assert torch.allclose(xr, x, atol=1e-3)

    assert torch.allclose(log_det_forward0, -log_det_inverse0, atol=1e-3)
    assert torch.allclose(log_det_forward1, -log_det_inverse1, atol=1e-3)
    assert torch.allclose(log_det_forward2, -log_det_inverse2, atol=1e-3)
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3)
