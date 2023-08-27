import pytest
import torch

from normalizing_flows import Flow
from normalizing_flows.src.bijections import DSCoupling, CouplingDSF, InverseDSCoupling, InverseCouplingDSF
from normalizing_flows.src.bijections.finite.autoregressive.transformers.combination import SigmoidTransform, \
    DeepSigmoidNetwork, InverseDeepSigmoidNetwork, InverseSigmoidTransform


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(1,), (5,), (100,), (3, 56, 2)])
def test_basic(event_shape, batch_shape):
    torch.manual_seed(0)

    forward_transformer = SigmoidTransform(event_shape=torch.Size(event_shape))
    inverse_transformer = InverseSigmoidTransform(event_shape=torch.Size(event_shape))

    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, forward_transformer.hidden_dim * 3))

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
def test_deep_sigmoid_network(event_shape, batch_shape, hidden_dim):
    torch.manual_seed(0)

    forward_transformer = DeepSigmoidNetwork(torch.Size(event_shape), hidden_dim=hidden_dim)
    inverse_transformer = InverseDeepSigmoidNetwork(torch.Size(event_shape), hidden_dim=hidden_dim)

    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, hidden_dim * len(forward_transformer.components) * 3))

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
@pytest.mark.parametrize('event_shape', [(2,), (5,), (100,), (3, 56, 2)])
def test_ds_coupling(event_shape, batch_shape):
    torch.manual_seed(0)

    forward_layer = DSCoupling(torch.Size(event_shape))
    inverse_layer = InverseDSCoupling(torch.Size(event_shape))

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
def test_coupling_dsf(n_dim, batch_shape):
    torch.manual_seed(0)

    forward_flow = Flow(CouplingDSF(n_dim))
    x = torch.randn(size=(*batch_shape, n_dim))
    log_prob = forward_flow.log_prob(x)

    assert log_prob.shape == batch_shape
    assert torch.all(~torch.isnan(log_prob))
    assert torch.all(~torch.isinf(log_prob))

    inverse_flow = Flow(InverseCouplingDSF(n_dim))
    x_new = inverse_flow.sample(len(x))

    assert x_new.shape == (len(x), *inverse_flow.bijection.event_shape)
    assert torch.all(~torch.isnan(x_new))
    assert torch.all(~torch.isinf(x_new))
