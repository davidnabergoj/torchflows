import pytest
import torch
import torch.nn as nn

from normalizing_flows.src.bijections.finite.residual.log_abs_det_estimators import log_det_hutchinson, log_det_roulette


@pytest.mark.parametrize('n_iterations', [4, 10, 25, 100])
def test_hutchinson(n_iterations):
    # an example of a Lipschitz continuous function with constant < 1: g(x) = 1/2 * x

    n_data = 100
    n_dim = 30

    class TestFunction(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            return 0.5 * inputs

    def jac_f(inputs):
        return torch.eye(n_dim) * 1.5

    def log_det_jac_f(inputs):
        return torch.log(torch.abs(torch.det(jac_f(inputs))))

    g = TestFunction()

    torch.manual_seed(0)
    x = torch.randn(size=(n_data, n_dim))
    g_value, log_det_f = log_det_hutchinson(g, x, training=False, n_iterations=n_iterations)
    log_det_f_true = log_det_jac_f(x).ravel()

    print(f'{log_det_f = }')
    print(f'{log_det_f_true = }')
    print(f'{log_det_f.mean() = }')
    assert torch.allclose(log_det_f, log_det_f_true)


@pytest.mark.parametrize('p', [0.1, 0.5, 0.9])
def test_roulette(p):
    # an example of a Lipschitz continuous function with constant < 1: g(x) = 1/2 * x

    n_data = 100
    n_dim = 30

    class TestFunction(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            return 0.5 * inputs

    def jac_f(inputs):
        return torch.eye(n_dim) * 1.5

    def log_det_jac_f(inputs):
        return torch.log(torch.abs(torch.det(jac_f(inputs))))

    g = TestFunction()

    torch.manual_seed(0)
    x = torch.randn(size=(n_data, n_dim))
    g_value, log_det_f = log_det_roulette(g, x, training=False, p=p)
    log_det_f_true = log_det_jac_f(x).ravel()

    print(f'{log_det_f = }')
    print(f'{log_det_f_true = }')
    print(f'{log_det_f.mean() = }')
    assert torch.allclose(log_det_f, log_det_f_true)
