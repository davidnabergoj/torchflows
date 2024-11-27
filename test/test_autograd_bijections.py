from typing import Tuple

import pytest
import torch

from torchflows import Flow
from torchflows.bijections.base import Bijection
from torchflows.bijections.finite.autoregressive.architectures import NICE, RealNVP, CouplingRQNSF, MAF, IAF, \
    InverseAutoregressiveRQNSF, MaskedAutoregressiveRQNSF
from torchflows.bijections.finite.autoregressive.layers import ElementwiseScale, ElementwiseAffine, ElementwiseShift, \
    LRSCoupling, LinearRQSCoupling, ElementwiseRQSpline
from torchflows.bijections.finite.matrix import HouseholderProductMatrix, LowerTriangularInvertibleMatrix, \
    UpperTriangularInvertibleMatrix, IdentityMatrix, RandomPermutationMatrix, ReversePermutationMatrix, QRMatrix, \
    LUMatrix
from torchflows.bijections.finite.residual.architectures import InvertibleResNet, ResFlow, ProximalResFlow
from torchflows.utils import get_batch_shape
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


def assert_valid_log_probability_gradient(bijection: Bijection, x: torch.Tensor, context: torch.Tensor):
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
    ElementwiseScale,
    ElementwiseAffine,
    ElementwiseShift,
    ElementwiseRQSpline
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_elementwise(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    IdentityMatrix,
    RandomPermutationMatrix,
    ReversePermutationMatrix,
    LowerTriangularInvertibleMatrix,
    UpperTriangularInvertibleMatrix,
    HouseholderProductMatrix,
    QRMatrix,
    LUMatrix,
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_matrix(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
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
def test_coupling(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
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
def test_masked_autoregressive(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple,
                               context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    InvertibleResNet,
    ResFlow,
    ProximalResFlow,
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_residual(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_log_probability_gradient(bijection, x, context)
