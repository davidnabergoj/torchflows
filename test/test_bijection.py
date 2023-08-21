from typing import Tuple

import pytest
import torch

from normalizing_flows.src.bijections.finite.linear import LU, Permutation, InverseLU, LowerTriangular, \
    HouseholderOrthogonal, QR
from normalizing_flows import RealNVP, MAF

@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (100,), (5, 1, 6, 7), (3, 13, 8)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (50, 50)])
@pytest.mark.parametrize('bijection_class', [
    LU,
    Permutation,
    InverseLU,
    LowerTriangular,
    HouseholderOrthogonal,
    QR
])
def test_basic(batch_shape: Tuple, event_shape: Tuple, bijection_class):
    # Event shape cannot be too big, otherwise
    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape)
    bij = bijection_class(event_shape=event_shape)
    z, log_det_forward = bij.forward(x)
    xr, log_det_inverse = bij.inverse(z)

    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr, atol=1e-3), f"{torch.max(torch.abs(x-xr)) = }"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3), \
        f"{torch.max(torch.abs(log_det_forward+log_det_inverse)) = }"

@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (100,), (5, 1, 6, 7), (3, 13, 8)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (50, 50)])
@pytest.mark.parametrize('bijection_class', [
    RealNVP,
    MAF
])
def test_neural_network(batch_shape: Tuple, event_shape: Tuple, bijection_class):
    # Event shape cannot be too big, otherwise
    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape)
    bij = bijection_class(event_shape)
    z, log_det_forward = bij.forward(x)
    xr, log_det_inverse = bij.inverse(z)

    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr, atol=1e-2), f"{torch.max(torch.abs(x-xr)) = }"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-2), \
        f"{torch.max(torch.abs(log_det_forward+log_det_inverse)) = }"
