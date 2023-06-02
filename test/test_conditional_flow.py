import pytest
import torch

from src import RealNVP, Flow


@pytest.mark.parametrize('n_dim', [2, 10, 100])
@pytest.mark.parametrize('context_dim', [1, 2, 10, 100])
def test_basic(n_dim, context_dim):
    n_data = 100
    flow = Flow(n_dim=n_dim, bijection=RealNVP(n_dim=n_dim))
    x = torch.randn(n_data, n_dim)
    c = torch.randn(n_data, context_dim)

    log_prob = flow.log_prob(x, context=c)
    assert log_prob.shape == (n_data,)

    x_new = flow.sample(5, context=c)
    assert x_new.shape == (5, n_data, n_dim)

    x_new_no_context = flow.sample(5)
    assert x_new_no_context.shape == (5, n_dim)
