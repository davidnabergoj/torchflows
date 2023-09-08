from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear import Linear as LinearSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear_rational import \
    LinearRational as LinearRationalSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import \
    RationalQuadratic as RationalQuadraticSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.cubic import Cubic as CubicSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.basis import Basis as BasisSpline


def setup_transformer_data(transformer_class: Transformer, batch_shape, event_shape):
    torch.manual_seed(0)
    transformer = transformer_class(event_shape)
    x = torch.randn(*batch_shape, *event_shape)
    h = torch.randn(*batch_shape, *event_shape, transformer.n_parameters)
    return transformer, x, h


def assert_valid_reconstruction(transformer: Transformer,
                                x: torch.Tensor,
                                h: torch.Tensor,
                                reconstruction_eps=1e-3,
                                log_det_eps=1e-3):
    z, log_det_forward = transformer.forward(x, h)
    xr, log_det_inverse = transformer.inverse(z, h)

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


@pytest.mark.parametrize('transformer_class', [
    LinearSpline,
    LinearRationalSpline,
    RationalQuadraticSpline,
    CubicSpline,
    BasisSpline
])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)])
@pytest.mark.parametrize('event_shape', [(2,), (3,), (2, 4), (100,), (3, 7, 2)])
def test_spline(transformer_class: Transformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)
