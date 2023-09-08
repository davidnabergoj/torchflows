import pytest
import torch

from normalizing_flows import Flow
from normalizing_flows.bijections import DSCoupling, CouplingDSF
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid import Sigmoid, DeepSigmoid, \
    DeepDenseSigmoid
from normalizing_flows.bijections.finite.base import invert


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(5,), (1,), (100,), (3, 56, 2)])
def test_sigmoid_transformer(event_shape, batch_shape):
    torch.manual_seed(0)

    forward_transformer = Sigmoid(event_shape=torch.Size(event_shape))
    inverse_transformer = invert(Sigmoid(event_shape=torch.Size(event_shape)))

    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, forward_transformer.n_parameters))

    y, log_det_forward = forward_transformer.forward(x, h)

    assert y.shape == x.shape
    assert log_det_forward.shape == batch_shape
    assert torch.all(~torch.isnan(y))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isinf(y))
    assert torch.all(~torch.isinf(log_det_forward))

    z = x
    x, log_det_inverse = inverse_transformer.inverse(z, h)

    assert z.shape == x.shape
    assert log_det_inverse.shape == batch_shape
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det_inverse))
    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(log_det_inverse))


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(1,), (5,), (100,), (3, 56, 2)])
@pytest.mark.parametrize('hidden_dim', [1, 2, 4, 8, 16, 32])
def test_deep_sigmoid_transformer(event_shape, batch_shape, hidden_dim):
    torch.manual_seed(0)

    forward_transformer = DeepSigmoid(torch.Size(event_shape), hidden_dim=hidden_dim)
    inverse_transformer = invert(DeepSigmoid(torch.Size(event_shape), hidden_dim=hidden_dim))

    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, forward_transformer.n_parameters))

    y, log_det_forward = forward_transformer.forward(x, h)

    assert y.shape == x.shape
    assert log_det_forward.shape == batch_shape
    assert torch.all(~torch.isnan(y))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isinf(y))
    assert torch.all(~torch.isinf(log_det_forward))

    z = x
    x, log_det_inverse = inverse_transformer.inverse(x, h)

    assert z.shape == x.shape
    assert log_det_inverse.shape == batch_shape
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det_inverse))
    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(log_det_inverse))


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(3, 56, 2), (2,), (5,), (100,)])
def test_deep_sigmoid_coupling(event_shape, batch_shape):
    torch.manual_seed(0)

    forward_layer = DSCoupling(torch.Size(event_shape))
    inverse_layer = invert(DSCoupling(torch.Size(event_shape)))

    x = torch.randn(size=(*batch_shape, *event_shape))
    y, log_det_forward = forward_layer.forward(x)

    assert y.shape == x.shape
    assert log_det_forward.shape == batch_shape
    assert torch.all(~torch.isnan(y))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isinf(y))
    assert torch.all(~torch.isinf(log_det_forward))

    z = x
    x, log_det_inverse = inverse_layer.inverse(z)

    assert z.shape == x.shape
    assert log_det_inverse.shape == batch_shape
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(log_det_inverse))
    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(log_det_inverse))


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('n_dim', [2, 5, 100, 1000])
def test_deep_sigmoid_coupling_flow(n_dim, batch_shape):
    torch.manual_seed(0)

    event_shape = torch.Size((n_dim,))

    forward_flow = Flow(CouplingDSF(event_shape))
    x = torch.randn(size=(*batch_shape, n_dim))
    log_prob = forward_flow.log_prob(x)

    assert log_prob.shape == batch_shape
    assert torch.all(~torch.isnan(log_prob))
    assert torch.all(~torch.isinf(log_prob))

    inverse_flow = Flow(invert(CouplingDSF(event_shape)))
    x_new = inverse_flow.sample(len(x))

    assert x_new.shape == (len(x), *inverse_flow.bijection.event_shape)
    assert torch.all(~torch.isnan(x_new))
    assert torch.all(~torch.isinf(x_new))
