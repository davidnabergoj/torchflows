from typing import Tuple

import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import ScalarTransformer
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear import Linear as LinearSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.linear_rational import \
    LinearRational as LinearRationalSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.spline.rational_quadratic import \
    RationalQuadratic as RationalQuadraticSpline
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.convolution import Invertible1x1Convolution
from normalizing_flows.bijections.finite.autoregressive.transformers.linear.affine import Affine, Scale, Shift
from normalizing_flows.bijections.finite.autoregressive.transformers.combination.sigmoid import Sigmoid, DeepSigmoid, \
    DenseSigmoid, DeepDenseSigmoid
from normalizing_flows.bijections.finite.autoregressive.transformers.integration.unconstrained_monotonic_neural_network import \
    UnconstrainedMonotonicNeuralNetwork
from normalizing_flows.utils import get_batch_shape
from test.constants import __test_constants


def setup_transformer_data(transformer_class: ScalarTransformer, batch_shape, event_shape):
    # vector_to_vector: does the transformer map a vector to vector? Otherwise, it maps a scalar to scalar.
    torch.manual_seed(0)
    transformer = transformer_class(event_shape)
    x = torch.randn(*batch_shape, *event_shape)
    h = torch.randn(*batch_shape, *transformer.parameter_shape)
    return transformer, x, h


def assert_valid_reconstruction(transformer: ScalarTransformer,
                                x: torch.Tensor,
                                h: torch.Tensor,
                                reconstruction_eps: float = 1e-3,
                                log_det_eps: float = 1e-3):
    batch_shape = get_batch_shape(x, transformer.event_shape)

    z, log_det_forward = transformer.forward(x, h)
    assert x.shape == z.shape
    assert torch.all(~torch.isnan(z))
    assert torch.all(~torch.isinf(z))
    assert torch.all(~torch.isnan(log_det_forward))
    assert torch.all(~torch.isinf(log_det_forward))
    assert log_det_forward.shape == batch_shape

    xr, log_det_inverse = transformer.inverse(z, h)

    assert torch.all(~torch.isnan(xr))
    assert torch.all(~torch.isnan(log_det_inverse))
    assert torch.all(~torch.isinf(xr))
    assert torch.all(~torch.isinf(log_det_inverse))
    assert log_det_inverse.shape == batch_shape

    assert log_det_forward.shape == log_det_inverse.shape
    assert torch.allclose(x, xr, atol=reconstruction_eps), \
        f"E: {(x - xr).abs().max():.16f}"
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=log_det_eps), \
        f"E: {(log_det_forward + log_det_inverse).abs().max():.16f}"


@pytest.mark.parametrize('transformer_class', [
    Affine,
    Scale,
    Shift
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_affine(transformer_class: ScalarTransformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)


@pytest.mark.parametrize('transformer_class', [
    LinearSpline,
    LinearRationalSpline,
    RationalQuadraticSpline,
    # CubicSpline,
    # BasisSpline
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_spline(transformer_class: ScalarTransformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)


@pytest.mark.parametrize('transformer_class', [
    UnconstrainedMonotonicNeuralNetwork
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_integration(transformer_class: ScalarTransformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)


@pytest.mark.parametrize('transformer_class', [Sigmoid, DeepSigmoid])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_combination_basic(transformer_class: ScalarTransformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)


@pytest.mark.parametrize('transformer_class', [
    DenseSigmoid,
    DeepDenseSigmoid
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('event_shape', __test_constants['event_shape'])
def test_combination_vector_to_vector(transformer_class: ScalarTransformer, batch_shape: Tuple, event_shape: Tuple):
    transformer, x, h = setup_transformer_data(transformer_class, batch_shape, event_shape)
    assert_valid_reconstruction(transformer, x, h)


@pytest.mark.parametrize("batch_size", [2, 3, 5, 7, 1])
@pytest.mark.parametrize('image_shape', __test_constants['image_shape'])
def test_convolution(batch_size: int, image_shape: Tuple):
    torch.manual_seed(0)
    transformer = Invertible1x1Convolution(image_shape)

    *image_dimensions, n_channels = image_shape

    images = torch.randn(size=(batch_size, *image_shape))
    parameters = torch.randn(size=(batch_size, *image_dimensions, *transformer.parameter_shape))
    latent_images, log_det_forward = transformer.forward(images, parameters)
    reconstructed_images, log_det_inverse = transformer.inverse(latent_images, parameters)

    assert log_det_forward.shape == (batch_size,)
    assert log_det_inverse.shape == (batch_size,)
    assert torch.isfinite(log_det_forward).all()
    assert torch.isfinite(log_det_inverse).all()
    assert torch.allclose(log_det_forward, -log_det_inverse, atol=1e-3)

    assert latent_images.shape == images.shape
    assert reconstructed_images.shape == images.shape
    assert torch.isfinite(latent_images).all()
    assert torch.isfinite(reconstructed_images).all()
    rec_err = torch.max(torch.abs(latent_images - reconstructed_images))
    assert torch.allclose(latent_images, reconstructed_images, atol=1e-2), f"{rec_err = }"
