import pytest
import torch

from src import RealNVP, Flow


@pytest.mark.parametrize('n_dim', [2, 10, 100])
@pytest.mark.parametrize('context_dim', [1, 2, 10, 100])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_context(n_dim, context_dim, device: str):
    torch.manual_seed(0)
    device = torch.device(device)

    n_data = 100
    flow = Flow(n_dim=n_dim, bijection=RealNVP(n_dim=n_dim, context_shape=(context_dim,)).to(device), device=device)
    x = torch.randn(n_data, n_dim)
    c = torch.randn(n_data, context_dim)

    log_prob = flow.log_prob(x.to(device), context=c.to(device))
    assert log_prob.shape == (n_data,)

    x_new = flow.sample(5, context=c.to(device))
    assert x_new.shape == (5, n_data, n_dim)

    with pytest.raises(RuntimeError):
        x_new_no_context = flow.sample(5)


@pytest.mark.parametrize('n_dim', [2, 10, 100])
def test_no_context(n_dim):
    torch.manual_seed(0)

    n_data = 100
    flow = Flow(n_dim=n_dim, bijection=RealNVP(n_dim=n_dim))
    x = torch.randn(n_data, n_dim)

    log_prob = flow.log_prob(x)
    assert log_prob.shape == (n_data,)

    x_new = flow.sample(5)
    assert x_new.shape == (5, n_dim)

    with pytest.raises(RuntimeError):
        flow.log_prob(x, context=torch.randn_like(x))

    with pytest.raises(RuntimeError):
        x_new_no_context = flow.sample(10, context=torch.randn_like(x))
