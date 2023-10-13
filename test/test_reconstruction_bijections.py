from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections import LU, ReversePermutation, LowerTriangular, \
    Orthogonal, QR, ElementwiseScale, LRSCoupling, LinearRQSCoupling
from normalizing_flows.bijections import RealNVP, MAF, CouplingRQNSF, MaskedAutoregressiveRQNSF, ResFlowBlock, \
    InvertibleResNetBlock, \
    ElementwiseAffine, ElementwiseShift, InverseAutoregressiveRQNSF, IAF, NICE
from normalizing_flows.bijections import FFJORD
from normalizing_flows.bijections.continuous.base import ContinuousBijection, ExactODEFunction
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.continuous.ddnf import DeepDiffeomorphicBijection
from normalizing_flows.bijections.continuous.otflow import OTFlow
from normalizing_flows.bijections.continuous.rnode import RNODE
from normalizing_flows.bijections.finite.residual.architectures import ResFlow, InvertibleResNet, ProximalResFlow
from normalizing_flows.bijections.finite.residual.planar import Planar
from normalizing_flows.bijections.finite.residual.proximal import ProximalResFlowBlock
from normalizing_flows.bijections.finite.residual.radial import Radial
from normalizing_flows.bijections.finite.residual.sylvester import Sylvester
from normalizing_flows.utils import get_batch_shape
from test.constants import __test_constants


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
    torch.manual_seed(0)

    if context is None:
        # We need this check for transformers, as they do not receive context as an input.
        z, log_det_forward = bijection.forward(x)
        xr, log_det_inverse = bijection.inverse(z)
    else:
        z, log_det_forward = bijection.forward(x, context=context)
        xr, log_det_inverse = bijection.inverse(z, context=context)

    batch_shape = get_batch_shape(x, bijection.event_shape)

    assert x.shape == z.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert log_det_forward.shape == batch_shape

    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isnan(log_det_inverse))

    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(xr))
    assert torch.all(~torch.isinf(log_det_forward))
    assert torch.all(~torch.isinf(log_det_inverse))

    assert torch.allclose(x, xr, atol=reconstruction_eps), \
        f"E: {(x - xr).abs().max():.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=log_det_eps), \
        f"E: {(log_det_forward + log_det_inverse).abs().max():.16f}"


def assert_valid_reconstruction_continuous(bijection: ContinuousBijection,
                                           x: torch.Tensor,
                                           context: torch.Tensor,
                                           reconstruction_eps=1e-3,
                                           log_det_eps=1e-3):
    torch.manual_seed(0)

    if context is None:
        # We need this check for transformers, as they do not receive context as an input.
        z, log_det_forward = bijection.forward(x)
        if isinstance(bijection.f, ExactODEFunction):
            xr, log_det_inverse = bijection.inverse(z)
        else:
            xr, log_det_inverse = bijection.inverse(z, noise=bijection.f.hutch_noise)
    else:
        z, log_det_forward = bijection.forward(x, context=context)
        if isinstance(bijection.f, ExactODEFunction):
            xr, log_det_inverse = bijection.inverse(z, context=context)
        else:
            xr, log_det_inverse = bijection.inverse(z, context=context, noise=bijection.f.hutch_noise)

    batch_shape = get_batch_shape(x, bijection.event_shape)

    # Check shapes
    assert x.shape == z.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert log_det_forward.shape == batch_shape
    assert bijection.regularization().shape == ()

    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isnan(log_det_inverse))
    assert torch.all(~torch.isnan(bijection.regularization()))

    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isinf(xr))
    assert torch.all(~torch.isinf(log_det_forward))
    assert torch.all(~torch.isinf(log_det_inverse))
    assert torch.all(~torch.isinf(bijection.regularization()))

    assert torch.allclose(x, xr, atol=reconstruction_eps), \
        f"E: {(x - xr).abs().max():.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=log_det_eps), \
        f"E: {(log_det_forward + log_det_inverse).abs().max():.16f}"


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
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
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
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_coupling(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    MAF,
    IAF,
    InverseAutoregressiveRQNSF,
    MaskedAutoregressiveRQNSF
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_masked_autoregressive(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple,
                               context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.skip(reason="Computation takes too long")
@pytest.mark.parametrize('bijection_class', [
    ProximalResFlowBlock,
    InvertibleResNetBlock,
    ResFlowBlock,
    ProximalResFlow,
    InvertibleResNet,
    ResFlow,
    Planar,
    Radial,
    Sylvester
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_residual(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    FFJORD,
    RNODE,
    OTFlow,
    # DeepDiffeomorphicBijection,  # Skip, reason: reconstruction fails due to the Euler integrator as proposed in the
    #                                      original method. Replacing the Euler integrator with RK4 fixes the issue.
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_continuous(bijection_class: ContinuousBijection, batch_shape: Tuple, event_shape: Tuple,
                    context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction_continuous(bijection, x, context)
