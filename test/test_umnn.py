from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.combination import \
    UnconstrainedMonotonicNeuralNetwork


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (100,), (5, 1, 6, 7), (3, 13, 8)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (25,)])
def test_umnn(batch_shape: Tuple, event_shape: Tuple):
    # Event shape cannot be too big, otherwise
    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape) / 100
    h = [
        torch.randn(*batch_shape, *event_shape, 20, 1 + 1),
        torch.randn(*batch_shape, *event_shape, 20, 20 + 1),
        torch.randn(*batch_shape, *event_shape, 20, 20 + 1),
        torch.randn(*batch_shape, *event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)
    z, log_det_forward = bij.forward(x, h.repeat(batch_shape[0] * event_shape[0], 1))
    xr, log_det_inverse = bij.inverse(z, h.repeat(batch_shape[0] * event_shape[0], 1))

    # FIXME h is reshaped incorrectly

    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == batch_shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr, atol=1e-3), f"{torch.max(torch.abs(x-xr)) = }"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3), \
        f"{torch.max(torch.abs(log_det_forward+log_det_inverse)) = }"


def test_umnn_forward():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_forward0 = bij.forward(x0, h)
    z1, log_det_forward1 = bij.forward(x1, h)
    z, log_det_forward = bij.forward(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1]), log_det_forward)


def test_umnn_inverse():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_inverse0 = bij.inverse(x0, h)
    z1, log_det_inverse1 = bij.inverse(x1, h)
    z, log_det_inverse = bij.inverse(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_inverse0, log_det_inverse1]), log_det_inverse)


def test_umnn_reconstruction():
    torch.manual_seed(0)
    event_shape = (1,)

    x0 = torch.tensor([-1.0]).view(1, 1)
    x1 = torch.tensor([1.0]).view(1, 1)
    x = torch.cat([x0, x1]).view(2, 1)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_forward0 = bij.forward(x0, h)
    z1, log_det_forward1 = bij.forward(x1, h)
    z, log_det_forward = bij.forward(x, h.repeat(2, 1))

    assert torch.allclose(torch.cat([z0, z1]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1]), log_det_forward)

    x0r, log_det_inverse0 = bij.inverse(z0, h)
    x1r, log_det_inverse1 = bij.inverse(z1, h)
    xr, log_det_inverse = bij.inverse(z, h.repeat(2, 1))

    assert torch.allclose(x0r, x0, atol=1e-4)
    assert torch.allclose(x1r, x1, atol=1e-4)
    assert torch.allclose(xr, x, atol=1e-4)

    assert torch.allclose(log_det_forward0, -log_det_inverse0, atol=1e-4)
    assert torch.allclose(log_det_forward1, -log_det_inverse1, atol=1e-4)
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-4)


def test_umnn_forward_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2)
    x1 = torch.tensor([1.0, 2.0]).view(1, 2)
    x2 = torch.tensor([5.0, 6.0]).view(1, 2)
    x = torch.cat([x0, x1, x2]).view(3, 2)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_forward0 = bij.forward(x0, h)
    z1, log_det_forward1 = bij.forward(x1, h)
    z2, log_det_forward2 = bij.forward(x2, h)
    z, log_det_forward = bij.forward(x, h.repeat(3, 1))

    assert torch.allclose(torch.cat([z0, z1, z2]), z)
    assert torch.allclose(torch.as_tensor([log_det_forward0, log_det_forward1, log_det_forward2]), log_det_forward)


def test_umnn_inverse_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2)
    x1 = torch.tensor([1.0, 2.0]).view(1, 2)
    x2 = torch.tensor([5.0, 6.0]).view(1, 2)
    x = torch.cat([x0, x1, x2]).view(3, 2)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_inverse0 = bij.inverse(x0, h)
    z1, log_det_inverse1 = bij.inverse(x1, h)
    z2, log_det_inverse2 = bij.inverse(x2, h)
    z, log_det_inverse = bij.inverse(x, h.repeat(3, 1))

    assert torch.allclose(torch.cat([z0, z1, z2]), z)
    assert torch.allclose(torch.as_tensor([log_det_inverse0, log_det_inverse1, log_det_inverse2]), log_det_inverse)


def test_umnn_reconstruction_large_event():
    torch.manual_seed(0)
    event_shape = (2,)

    x0 = torch.tensor([-1.0, -2.0]).view(1, 2) / 10
    x1 = torch.tensor([1.0, 2.0]).view(1, 2) / 10
    x2 = torch.tensor([5.0, 6.0]).view(1, 2) / 10
    x = torch.cat([x0, x1, x2]).view(3, 2)

    h = [
        torch.randn(*event_shape, 20, 1 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 20, 20 + 1),
        torch.randn(*event_shape, 1, 20 + 1),
    ]
    h = torch.cat([torch.flatten(e, start_dim=len(event_shape)) for e in h], dim=1)
    bij = UnconstrainedMonotonicNeuralNetwork(event_shape=event_shape, n_hidden_layers=2, hidden_dim=20)

    z0, log_det_forward0 = bij.forward(x0, h)
    z1, log_det_forward1 = bij.forward(x1, h)
    z2, log_det_forward2 = bij.forward(x2, h)
    z, log_det_forward = bij.forward(x, h.repeat(3, 1))

    x0r, log_det_inverse0 = bij.inverse(z0, h)
    x1r, log_det_inverse1 = bij.inverse(z1, h)
    x2r, log_det_inverse2 = bij.inverse(z2, h)
    xr, log_det_inverse = bij.inverse(z, h.repeat(3, 1))

    assert torch.allclose(x0r, x0, atol=1e-3)
    assert torch.allclose(x1r, x1, atol=1e-3)
    assert torch.allclose(x2r, x2, atol=1e-3)
    assert torch.allclose(xr, x, atol=1e-3)

    assert torch.allclose(log_det_forward0, -log_det_inverse0, atol=1e-3)
    assert torch.allclose(log_det_forward1, -log_det_inverse1, atol=1e-3)
    assert torch.allclose(log_det_forward2, -log_det_inverse2, atol=1e-3)
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3)
