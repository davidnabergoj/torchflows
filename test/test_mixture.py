from torchflows.flows import FlowMixture, Flow
from torchflows.bijections.finite.autoregressive import RealNVP, NICE, CouplingRQNSF
import torch


def test_basic():
    torch.manual_seed(0)

    n_data = 100
    n_dim = 10
    x = torch.randn(size=(n_data, n_dim))

    mixture = FlowMixture([
        Flow(RealNVP(event_shape=(n_dim,))),
        Flow(NICE(event_shape=(n_dim,))),
        Flow(CouplingRQNSF(event_shape=(n_dim,)))
    ])

    log_prob = mixture.log_prob(x)
    assert log_prob.shape == (n_data,)
    assert torch.all(torch.isfinite(log_prob))

    x_sampled = mixture.sample(n_data)
    assert x_sampled.shape == x.shape
    assert torch.all(torch.isfinite(x_sampled))


def test_medium():
    torch.manual_seed(0)

    n_data = 1000
    n_dim = 100
    x = torch.randn(size=(n_data, n_dim))

    mixture = FlowMixture([
        Flow(RealNVP(event_shape=(n_dim,))),
        Flow(NICE(event_shape=(n_dim,))),
        Flow(CouplingRQNSF(event_shape=(n_dim,)))
    ])

    log_prob = mixture.log_prob(x)
    assert log_prob.shape == (n_data,)
    assert torch.all(torch.isfinite(log_prob))

    x_sampled = mixture.sample(n_data)
    assert x_sampled.shape == x.shape
    assert torch.all(torch.isfinite(x_sampled))


def test_complex_event():
    torch.manual_seed(0)

    n_data = 1000
    event_shape = (2, 3, 4, 5)
    x = torch.randn(size=(n_data, *event_shape))

    mixture = FlowMixture([
        Flow(RealNVP(event_shape=event_shape)),
        Flow(NICE(event_shape=event_shape)),
        Flow(CouplingRQNSF(event_shape=event_shape))
    ])

    log_prob = mixture.log_prob(x)
    assert log_prob.shape == (n_data,)
    assert torch.all(torch.isfinite(log_prob))

    x_sampled = mixture.sample(n_data)
    assert x_sampled.shape == x.shape
    assert torch.all(torch.isfinite(x_sampled))
