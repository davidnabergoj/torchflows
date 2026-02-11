from typing import Tuple

import pytest
import torch

from torchflows.bijections.continuous.base import ContinuousBijection
from torchflows.bijections.base import Bijection
from torchflows.bijections.continuous.ddnf import DDNF
from torchflows.bijections.continuous.ffjord import FFJORD
from torchflows.bijections.continuous.otflow import OTFlowBijection
from torchflows.bijections.continuous.rnode import RNODE
from torchflows.bijections.finite.autoregressive.architectures import NICE, RealNVP, CouplingRQNSF, MAF, IAF, \
    InverseAutoregressiveRQNSF, MaskedAutoregressiveRQNSF
from torchflows.bijections.finite.autoregressive.layers import ElementwiseScale, ElementwiseAffine, ElementwiseShift, \
    LRSCoupling, LinearRQSCoupling, ActNorm, DenseSigmoidalCoupling, DeepDenseSigmoidalCoupling, DeepSigmoidalCoupling
from torchflows.bijections.finite.matrix import LUMatrix, ReversePermutationMatrix, LowerTriangularInvertibleMatrix, \
    HouseholderProductMatrix, QRMatrix
from torchflows.bijections.finite.residual.architectures import ResFlow, InvertibleResNet, ProximalResFlow
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
    context = torch.randn(
        size=(*batch_shape, *context_shape)
    ) if context_shape is not None else None


    if bijection_class in [DDNF, FFJORD, RNODE, OTFlowBijection]:
        b_atol = 1e-5
        b_rtol = 1e-5
        solver = 'dopri5'
        if bijection_class != OTFlowBijection:
            # dopri5?
            bijection = bijection_class(
                event_shape,
                context_shape=context_shape,
                reuse_noise=True,
                solver=solver,
                atol=b_atol,
                rtol=b_rtol,
            )
        else:
            bijection = bijection_class(
                event_shape,
                context_shape=context_shape,
                atol=b_atol,
                rtol=b_rtol,
                solver=solver
            )
        bijection.eval()  # Ignore regularization
    else:
        bijection = bijection_class(event_shape, context_shape=context_shape)
    return bijection, x, context


def assert_valid_reconstruction(bijection: Bijection,
                                x: torch.Tensor,
                                context: torch.Tensor,
                                reconstruction_eps: float = 1e-3,
                                log_det_eps: float = 1e-3):
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

    assert z.isfinite().all()
    assert xr.isfinite().all()
    assert log_det_forward.isfinite().all()
    assert log_det_inverse.isfinite().all()

    assert torch.allclose(x, xr, atol=reconstruction_eps), \
        f"E: {(x - xr).abs().max():.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=log_det_eps), \
        f"E: {(log_det_forward + log_det_inverse).abs().max():.16f}"


def assert_valid_reconstruction_continuous(bijection: ContinuousBijection,
                                           x: torch.Tensor,
                                           context: torch.Tensor,
                                           rtol_rec: float = 1e-1,
                                           rtol_log_det: float = 1e-1):
    torch.manual_seed(0)

    z, log_det_forward = bijection.forward(x, context=context)
    xr, log_det_inverse = bijection.inverse(z, context=context)

    batch_shape = get_batch_shape(x, bijection.event_shape)

    # Check shapes
    assert x.shape == z.shape
    assert log_det_forward.shape == log_det_inverse.shape
    assert log_det_forward.shape == batch_shape
    assert bijection.regularization().shape == ()

    assert z.isfinite().all()
    assert xr.isfinite().all()
    assert log_det_forward.isfinite().all()
    assert log_det_inverse.isfinite().all()
    assert bijection.regularization().isfinite().all()

    assert torch.allclose(x, xr, rtol=rtol_rec), \
        f"E: {(x - xr).abs().max():.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=rtol_log_det), \
        f"E: {(log_det_forward + log_det_inverse).abs().max():.16f}"


@pytest.mark.parametrize('bijection_class', [
    LUMatrix,
    ReversePermutationMatrix,
    ElementwiseScale,
    LowerTriangularInvertibleMatrix,
    HouseholderProductMatrix,
    QRMatrix,
    ElementwiseAffine,
    ElementwiseShift,
    ActNorm
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_linear(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(
        bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    NICE,
    RealNVP,
    CouplingRQNSF,
    LRSCoupling,
    LinearRQSCoupling,
    DenseSigmoidalCoupling,
    DeepDenseSigmoidalCoupling,
    DeepSigmoidalCoupling,
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_coupling(bijection_class: Bijection, batch_shape: Tuple, event_shape: Tuple, context_shape: Tuple):
    bijection, x, context = setup_data(
        bijection_class, batch_shape, event_shape, context_shape)
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
    bijection, x, context = setup_data(
        bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.skip(reason="Computation takes too long / inherently inaccurate")
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
    bijection, x, context = setup_data(
        bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction(bijection, x, context)


@pytest.mark.parametrize('bijection_class', [
    FFJORD,
    RNODE,
    OTFlowBijection,
    # DDNF # Skip, reason: reconstruction fails due to the Euler integrator as proposed in the
    #                                     original method. Replacing the Euler integrator with RK4 fixes the issue.
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
@pytest.mark.local_only  # Too slow with DOPRI5
def test_continuous(
        bijection_class: ContinuousBijection,
        batch_shape: Tuple,
        event_shape: Tuple,
        context_shape: Tuple
):
    bijection, x, context = setup_data(
        bijection_class, batch_shape, event_shape, context_shape)
    assert_valid_reconstruction_continuous(bijection, x, context)
