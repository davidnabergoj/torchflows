import torch

from torchflows.bijections.finite.autoregressive.transformers.linear.matrix import LUTransformer
from test.constants import __test_constants


def test_basic():
    torch.manual_seed(0)

    batch_shape = (2, 3)
    event_shape = (5, 7)

    transformer = LUTransformer(event_shape)

    x = torch.randn(size=(*batch_shape, *event_shape))
    h = torch.randn(size=(*batch_shape, *transformer.parameter_shape))

    z, log_det_forward = transformer.forward(x, h)
    x_reconstructed, log_det_inverse = transformer.inverse(z, h)

    assert torch.allclose(
        x,
        x_reconstructed,
        atol=__test_constants['data_atol']
    ), f"{torch.linalg.norm(x - x_reconstructed)}"
    assert torch.allclose(
        log_det_forward,
        -log_det_inverse,
        atol=__test_constants['log_det_atol']
    )
