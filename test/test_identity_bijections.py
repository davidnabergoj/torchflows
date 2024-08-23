# Check that when all bijection parameters are set to 0, the bijections reduce to an identity map

from torchflows.bijections.finite.autoregressive.layers import (
    AffineCoupling,
    DeepSigmoidalCoupling,
    RQSCoupling,
    InverseAffineCoupling,
    LRSCoupling,
    ShiftCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    ElementwiseAffine,
    ElementwiseRQSpline,
    ElementwiseScale,
    ElementwiseShift,
    LinearAffineCoupling,
    LinearLRSCoupling,
    LinearRQSCoupling,
    LinearShiftCoupling,
    LRSForwardMaskedAutoregressive,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive,
    UMNNMaskedAutoregressive,

)
import torch
import pytest
from test.constants import __test_constants


@pytest.mark.parametrize(
    'layer_class',
    [
        AffineCoupling,
        DeepSigmoidalCoupling,
        RQSCoupling,
        InverseAffineCoupling,
        LRSCoupling,
        ShiftCoupling,
        AffineForwardMaskedAutoregressive,
        AffineInverseMaskedAutoregressive,
        ElementwiseAffine,
        ElementwiseRQSpline,
        ElementwiseScale,
        ElementwiseShift,
        LinearAffineCoupling,
        LinearLRSCoupling,
        LinearRQSCoupling,
        LinearShiftCoupling,
        LRSForwardMaskedAutoregressive,
        RQSForwardMaskedAutoregressive,
        RQSInverseMaskedAutoregressive,
        # UMNNMaskedAutoregressive,  # Inexact due to numerics
    ]
)
def test_basic(layer_class):
    n_batch, n_dim = 2, 3

    torch.manual_seed(0)
    x = torch.randn(size=(n_batch, n_dim))
    layer = layer_class(event_shape=torch.Size((n_dim,)))

    # Set all conditioner parameters to 0
    with torch.no_grad():
        for p in layer.parameters():
            p.data *= 0

    assert torch.allclose(layer(x)[0], x, atol=__test_constants['data_atol_easy'])  # 1e-2
