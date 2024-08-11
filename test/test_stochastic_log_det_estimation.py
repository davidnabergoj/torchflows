import pytest
import torch
import torch.nn as nn

from normalizing_flows.bijections.finite.residual.log_abs_det_estimators import log_det_power_series, log_det_roulette
from test.constants import __test_constants


# TODO fix tests: replace closeness checks with analytic bounds

class LipschitzTestData:
    def __init__(self, n_dim):
        self.n_dim = n_dim

    class LipschitzFunction(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            return 0.5 * inputs

    def jac_f(self, _):
        return torch.eye(self.n_dim) * (1 + 0.5)

    def log_det_jac_f(self, inputs):
        return torch.log(torch.abs(torch.det(self.jac_f(inputs))))


@pytest.mark.skip(reason="Improper check")
@pytest.mark.parametrize('n_hutchinson_samples', [*list(range(25, 40))])
@pytest.mark.parametrize('n_iterations', [4, 10, 25, 100])
def test_power_series_estimator(n_iterations, n_hutchinson_samples):
    # This test checks for validity of the hutchinson power series trace estimator.
    # The estimator computes log|det(Jac_f)| where f(x) = x + g(x) and x is Lipschitz continuous with Lip(g) < 1.
    # In this example: a Lipschitz continuous function with constant < 1 is g(x) = 1/2 * x; Lip(g) = 1/2.

    # The reference jacobian of f is I * 1.5, because d/dx f(x) = d/dx x + g(x) = d/dx x + 1/2 * x = 1 + 1/2 = 1.5

    # TODO: use the analytical variance of the Monte Carlo Hutchinson trace estimator to compute the variance of the
    #  Hutchinson power series estimator. Then make sure that the power series error is below 4 * variance.

    n_data = 1
    n_dim = 1

    test_data = LipschitzTestData(n_dim)
    g = test_data.LipschitzFunction()

    torch.manual_seed(0)
    x = torch.randn(size=(n_data, n_dim))
    g_value, log_det_f_estimated = log_det_power_series(
        g,
        x,
        training=False,
        n_iterations=n_iterations,
        n_hutchinson_samples=n_hutchinson_samples
    )
    log_det_f_true = test_data.log_det_jac_f(x).ravel()

    print()
    print(f'{log_det_f_estimated = }')
    print(f'{log_det_f_true = }')
    assert torch.allclose(log_det_f_estimated, log_det_f_true, atol=__test_constants['log_det_atol'])


@pytest.mark.skip(reason="Improper check")
@pytest.mark.parametrize('p', [0.01, 0.1, 0.5, 0.9, 0.99])
def test_roulette_estimator(p):
    # an example of a Lipschitz continuous function with constant < 1: g(x) = 1/2 * x

    n_data = 100
    n_dim = 30

    test_data = LipschitzTestData(n_dim)

    g = test_data.LipschitzFunction()

    torch.manual_seed(0)
    x = torch.randn(size=(n_data, n_dim))
    g_value, log_det_f = log_det_roulette(g, x, training=False, p=p)
    log_det_f_true = test_data.log_det_jac_f(x).ravel()

    print(f'{log_det_f = }')
    print(f'{log_det_f_true = }')
    print(f'{log_det_f.mean() = }')
    assert torch.allclose(log_det_f, log_det_f_true, atol=__test_constants['log_det_atol'])
