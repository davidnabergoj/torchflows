import math

import torch

from torchflows.base_distributions.gaussian import DiagonalGaussian


def test_standard_gaussian():
    x = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ])
    dist = DiagonalGaussian(loc=torch.tensor([0.0, 0.0, 0.0]), scale=torch.tensor([1.0, 1.0, 1.0]))
    log_prob_dist = dist.log_prob(x)
    log_prob_manual = -3 / 2 * math.log(2 * math.pi) - 1 / 2 * torch.tensor([3.0, 1.0, 5.0])

    assert torch.allclose(log_prob_dist, log_prob_manual)


def test_diagonal_gaussian():
    x = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ])
    dist = DiagonalGaussian(loc=torch.tensor([0.0, 0.0, 0.0]), scale=torch.tensor([1.0, 2.0, 3.0]))
    det_sigma = 1 * 4 * 9
    log_prob_dist = dist.log_prob(x)
    log_prob_manual = -3 / 2 * math.log(2 * math.pi) - 1 / 2 * math.log(det_sigma) - 1 / 2 * torch.tensor(
        [49 / 36, 1 / 9, 10 / 9])

    assert torch.allclose(log_prob_dist, log_prob_manual)
