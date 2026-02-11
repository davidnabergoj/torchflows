import pytest
import torch

from torchflows.flows import Flow
from torchflows.bijections.finite.autoregressive.architectures import CouplingDeepSF
from torchflows.bijections.finite.autoregressive.layers import DeepSigmoidalCoupling
from torchflows.bijections.finite.autoregressive.transformers.combination.sigmoid import Sigmoid, DeepSigmoid
from torchflows.bijections.base import invert
from test.constants import __test_constants


@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
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


@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
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


@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_deep_sigmoid_coupling(event_shape, batch_shape):
    torch.manual_seed(0)

    forward_layer = DeepSigmoidalCoupling(torch.Size(event_shape))
    inverse_layer = invert(DeepSigmoidalCoupling(torch.Size(event_shape)))

    x = torch.randn(size=(*batch_shape, *event_shape))  # Reduce magnitude for stability
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


@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_deep_sigmoid_coupling_flow(event_shape, batch_shape):
    torch.manual_seed(0)

    n_dim = int(torch.prod(torch.tensor(event_shape)))
    event_shape = (n_dim,)  # Overwrite

    forward_flow = Flow(CouplingDeepSF(event_shape))
    x = torch.randn(size=(*batch_shape, n_dim))
    log_prob = forward_flow.log_prob(x)

    assert log_prob.shape == batch_shape
    assert torch.all(~torch.isnan(log_prob))
    assert torch.all(~torch.isinf(log_prob))

    inverse_flow = Flow(invert(CouplingDeepSF(event_shape)))
    x_new = inverse_flow.sample(len(x))

    assert x_new.shape == (len(x), *inverse_flow.bijection.event_shape)
    assert torch.all(~torch.isnan(x_new))
    assert torch.all(~torch.isinf(x_new))
