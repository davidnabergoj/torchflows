from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections import LU, ReversePermutation, LowerTriangular, \
    Orthogonal, QR, ElementwiseScale, LRSCoupling, LinearRQSCoupling
from normalizing_flows.bijections import RealNVP, MAF, CouplingRQNSF, MaskedAutoregressiveRQNSF, ResFlow, \
    InvertibleResNet, \
    ElementwiseAffine, ElementwiseShift, InverseAutoregressiveRQNSF, IAF, NICE
from normalizing_flows.bijections.finite.base import Bijection


def setup_data(bijection_class, batch_shape, event_shape, context_shape):
    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape)
    if context_shape is not None:
        context = torch.randn(size=(*batch_shape, *context_shape))
    else:
        context = None
    bijection = bijection_class(event_shape)
    return bijection, x, context


def assert_valid_reconstruction(bijection: Bijection,
                                x: torch.Tensor,
                                context: torch.Tensor,
                                reconstruction_eps=1e-3,
                                log_det_eps=1e-3):
    if context is None:
        # We need this check for transformers, as they do not receive context as an input.
        z, log_det_forward = bijection.forward(x)
        xr, log_det_inverse = bijection.inverse(z)
    else:
        z, log_det_forward = bijection.forward(x, context=context)
        xr, log_det_inverse = bijection.inverse(z, context=context)

    assert x.shape == z.shape
    assert log_det_forward.shape == log_det_inverse.shape

    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(xr))
    assert torch.all(~torch.isinf(log_det_forward))
    assert torch.all(~torch.isinf(log_det_inverse))

    assert torch.allclose(x, xr, atol=reconstruction_eps), \
        f"E: {(x - xr).abs().max()[0]:.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=log_det_eps), \
        f"E: {(log_det_forward + log_det_inverse).abs().max()[0]:.16f}"


@pytest.mark.parametrize('bijection_class', [
    LU,
    ReversePermutation,
    ElementwiseScale,
    LowerTriangular,
    Orthogonal,
    QR,
    ElementwiseAffine,
    ElementwiseShift
])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
@pytest.mark.parametrize('context_shape', [None, (2,), (3,), (2, 4), (5,)])
def test_linear(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    NICE,
    RealNVP,
    CouplingRQNSF,
    LRSCoupling,
    LinearRQSCoupling
])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
@pytest.mark.parametrize('context_shape', [None, (2,), (3,), (2, 4), (5,)])
def test_coupling(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    MAF,
    IAF,
    InverseAutoregressiveRQNSF,
    MaskedAutoregressiveRQNSF
])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
@pytest.mark.parametrize('context_shape', [None, (2,), (3,), (2, 4), (5,)])
def test_masked_autoregressive(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple,
                               context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.skip(reason="Computation takes too long")
@pytest.mark.parametrize('bijection_class', [
    InvertibleResNet,
    ResFlow,
])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,)])
@pytest.mark.parametrize('context_shape', [None, (2,), (3,), (2, 4), (5,)])
def test_residual(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)
