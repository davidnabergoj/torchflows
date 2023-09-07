from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections import LU, ReversePermutation, LowerTriangular, \
    Orthogonal, QR, ElementwiseScale, LRSCoupling, LinearRQSCoupling
from normalizing_flows import RealNVP, MAF, CouplingRQNSF, MaskedAutoregressiveRQNSF, ResFlow, InvertibleResNet, \
    ElementwiseAffine, ElementwiseShift, InverseAutoregressiveRQNSF, IAF, NICE, Flow


@pytest.mark.parametrize('n_dim', [1, 2, 10, 100])
def test_permutation_reconstruction(n_dim):
    torch.manual_seed(0)
    x = torch.randn(25, n_dim)
    bijection = ReversePermutation(event_shape=torch.Size((n_dim,)))

    z, log_det_forward = bijection(x)

    assert log_det_forward.shape == (x.shape[0],)
    assert torch.allclose(log_det_forward, torch.zeros_like(log_det_forward))

    x_reconstructed, log_det_inverse = bijection.inverse(z)

    assert log_det_inverse.shape == (x.shape[0],)
    assert torch.allclose(log_det_inverse, torch.zeros_like(log_det_inverse))

    assert torch.allclose(x, x_reconstructed)
    assert torch.allclose(torch.unique(x), torch.unique(z))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
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
def test_linear_reconstruction(batch_shape: Tuple, event_shape: Tuple, bijection_class):
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


@pytest.mark.skip(reason="Computation takes too long")
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,)])
@pytest.mark.parametrize('bijection_class', [
    InvertibleResNet,
    ResFlow,
])
def test_residual_reconstruction(batch_shape: Tuple, event_shape: Tuple, bijection_class):
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


def test_maf_nontrivial_event_shape():
    batch_shape = (2, 4)
    event_shape = (3, 7)

    torch.manual_seed(0)
    x = torch.randn(*batch_shape, *event_shape)
    bij = MAF(event_shape)
    z, log_det_forward = bij.forward(x)
    xr, log_det_inverse = bij.inverse(z)

    assert x.shape == z.shape == xr.shape
    assert log_det_forward.shape == log_det_inverse.shape


def test_coupling_nontrivial_event_and_batch_shape():
    event_shape = torch.Size((2, 4))
    batch_shape = torch.Size((3, 5))
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))
    bijection = NICE(event_shape)
    bijection.forward(x)

@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
@pytest.mark.parametrize('bijection_class', [
    NICE,
    RealNVP,
    MAF,
    IAF,
    CouplingRQNSF,
    InverseAutoregressiveRQNSF,
    MaskedAutoregressiveRQNSF,
    LRSCoupling,
    LinearRQSCoupling
])
def test_autoregressive_reconstruction(batch_shape: Tuple, event_shape: Tuple, bijection_class):
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


@pytest.mark.parametrize('architecture_class', [
    NICE,
    RealNVP,
    MAF,
    IAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF,
    InverseAutoregressiveRQNSF,
    LRSCoupling,
    LinearRQSCoupling
])
@pytest.mark.parametrize('n_dim', [2, 3, 10, 20])
def test_autoregressive_backward(architecture_class, n_dim):
    torch.manual_seed(0)
    event_shape = torch.Size((n_dim,))
    bijection = architecture_class(event_shape=event_shape)
    flow = Flow(bijection=bijection)
    x = torch.randn(size=(125, n_dim)) * 5
    loss = -flow.log_prob(x).mean()
    loss.backward()
