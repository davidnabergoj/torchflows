import pytest
import torch

from normalizing_flows.src.bijections.finite.autoregressive.transformers.combination import SigmoidTransform


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(2,), (5,), (100,), (3, 56, 2)])
def test_shape(event_shape, batch_shape):
    torch.manual_seed(0)
    transformer = SigmoidTransform(event_shape=torch.Size(event_shape))
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, transformer.hidden_dim, 3))
    y, log_det = transformer.forward(x, h)
    assert y.shape == x.shape
    assert log_det.shape == batch_shape


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(2,), (5,), (100,), (3, 56, 2)])
def test_nan(event_shape, batch_shape):
    torch.manual_seed(0)
    transformer = SigmoidTransform(event_shape=torch.Size(event_shape))
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, transformer.hidden_dim, 3))
    y, log_det = transformer.forward(x, h)
    assert torch.all(~torch.isnan(y))
    assert torch.all(~torch.isnan(log_det))


@pytest.mark.parametrize('batch_shape', [(7,), (25,), (13,), (2, 37)])
@pytest.mark.parametrize('event_shape', [(2,), (5,), (100,), (3, 56, 2)])
def test_inf(event_shape, batch_shape):
    torch.manual_seed(0)
    transformer = SigmoidTransform(event_shape=torch.Size(event_shape))
    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *event_shape, transformer.hidden_dim, 3))
    y, log_det = transformer.forward(x, h)
    assert torch.all(~torch.isinf(y))
    assert torch.all(~torch.isinf(log_det))
