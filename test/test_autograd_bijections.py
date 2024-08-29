from typing import Tuple

import pytest
import torch

from torchflows import Flow
from torchflows.bijections.base import Bijection
from torchflows.bijections.finite.autoregressive.architectures import NICE, RealNVP, CouplingRQNSF, MAF, IAF, \
    InverseAutoregressiveRQNSF, MaskedAutoregressiveRQNSF
from torchflows.bijections.finite.autoregressive.layers import ElementwiseScale, ElementwiseAffine, ElementwiseShift, \
    LRSCoupling, LinearRQSCoupling
from torchflows.bijections.finite.linear import LU, ReversePermutation, LowerTriangular, Orthogonal, QR
from torchflows.bijections.finite.residual.architectures import InvertibleResNet, ResFlow, ProximalResFlow
from torchflows.bijections.finite.residual.iterative import InvertibleResNetBlock, ResFlowBlock
from torchflows.bijections.finite.residual.planar import Planar
from torchflows.bijections.finite.residual.proximal import ProximalResFlowBlock
from torchflows.bijections.finite.residual.radial import Radial
from torchflows.bijections.finite.residual.sylvester import Sylvester
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
