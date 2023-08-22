import pytest
import torch
import torch.nn as nn

from normalizing_flows.src.bijections.finite.residual.log_abs_det_estimators import log_det_hutchinson


@pytest.mark.parametrize('n_iterations', [4, 10, 25, 100])
def test_hutchinson(n_iterations):
    # an example of a Lipschitz continuous function with constant < 1: g(x) = 1/2 * x

    class TestFunction(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            return 0.5 * inputs

    def ground_truth_jac(inputs):
        return torch.as_tensor(3 / 2)

    def ground_truth_log_abs_det_jac(inputs):
        return torch.log(torch.abs(ground_truth_jac(inputs)))

    x = torch.tensor([
        [-3.0],
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    g = TestFunction()

    torch.manual_seed(0)
    g_value, log_det = log_det_hutchinson(g, x, training=False, n_iterations=n_iterations)
    # note that log_det refers to x + g(x)
    log_det_true = ground_truth_log_abs_det_jac(x).ravel()

    print(f'{log_det = }')
    print(f'{log_det_true = }')
    assert torch.allclose(log_det, log_det_true)
