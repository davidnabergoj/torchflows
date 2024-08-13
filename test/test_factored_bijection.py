import torch
from torchflows.bijections.finite.multiscale.base import FactoredBijection
from torchflows.bijections.finite.autoregressive.layers import ElementwiseAffine
from test.constants import __test_constants


def test_basic():
    torch.manual_seed(0)

    bijection = FactoredBijection(
        event_shape=(6, 6),
        small_bijection_event_shape=(3, 3),
        small_bijection_mask=torch.tensor([
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ]),
        small_bijection=ElementwiseAffine(event_shape=(3, 3))
    )

    x = torch.randn(100, *bijection.event_shape)
    z, log_det_forward = bijection.forward(x)

    assert torch.allclose(
        x[..., ~bijection.transformed_event_mask],
        z[..., ~bijection.transformed_event_mask],
        __test_constants['data_atol_easy']
    )

    assert ~torch.allclose(
        x[..., bijection.transformed_event_mask],
        z[..., bijection.transformed_event_mask]
    )

    xr, log_det_inverse = bijection.inverse(z)
    assert torch.allclose(x, xr, __test_constants['data_atol_easy'])
    assert torch.allclose(log_det_forward, -log_det_inverse, __test_constants['log_det_atol_easy'])
