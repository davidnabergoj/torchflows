from typing import Tuple

import pytest
import torch

from normalizing_flows import Flow
from normalizing_flows.bijections import LU, ReversePermutation, LowerTriangular, \
    Orthogonal, QR, ElementwiseScale, LRSCoupling, LinearRQSCoupling
from normalizing_flows.bijections import RealNVP, MAF, CouplingRQNSF, MaskedAutoregressiveRQNSF, ResFlow, \
    InvertibleResNet, \
    ElementwiseAffine, ElementwiseShift, InverseAutoregressiveRQNSF, IAF, NICE
from normalizing_flows.bijections.finite.base import ConditionalBijection
from normalizing_flows.bijections.finite.residual.planar import Planar
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


def assert_valid_log_probability_gradient(bijection: ConditionalBijection, x: torch.Tensor, context: torch.Tensor):
    xc = x.clone()
    xc.requires_grad_(True)
    log_prob = Flow(bijection).log_prob(xc, context=context)

    batch_shape = get_batch_shape(x, bijection.event_shape)
    assert log_prob.shape == batch_shape
    assert torch.all(~torch.isnan(log_prob))
    assert torch.all(~torch.isinf(log_prob))

    grad_log_prob = torch.autograd.grad(log_prob.mean(), xc)[0]
    assert grad_log_prob.shape == x.shape
    assert torch.all(~torch.isnan(grad_log_prob))
    assert torch.all(~torch.isinf(grad_log_prob))


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
def test_linear(bijection_class: ConditionalBijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)


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
def test_coupling(bijection_class: ConditionalBijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    MAF,
    IAF,
    InverseAutoregressiveRQNSF,
    MaskedAutoregressiveRQNSF
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_masked_autoregressive(bijection_class: ConditionalBijection, batch_shape: Tuple, event_shape: Tuple,
                               context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)


@pytest.mark.skip(reason="Computation takes too long")
@pytest.mark.parametrize('bijection_class', [
    InvertibleResNet,
    ResFlow,
    Planar,
    Radial,
    Sylvester
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_residual(bijection_class: ConditionalBijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)
