import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import (
    MADE, FeedForward, LinearMADE, ResidualFeedForward, Constant, Linear
)
from test.constants import __test_constants


@pytest.mark.parametrize('transform_class', [
    MADE,
    FeedForward,
    LinearMADE,
    ResidualFeedForward,
    Linear
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('input_event_shape', __test_constants['input_event_shape'])
@pytest.mark.parametrize('output_event_shape', __test_constants['output_event_shape'])
@pytest.mark.parametrize('n_predicted_parameters', __test_constants['n_predicted_parameters'])
def test_shape(transform_class, batch_shape, input_event_shape, output_event_shape, n_predicted_parameters):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *input_event_shape))
    transform = transform_class(
        input_event_shape=input_event_shape,
        output_event_shape=output_event_shape,
        parameter_shape=n_predicted_parameters
    )
    out = transform(x)
    assert out.shape == (*batch_shape, *output_event_shape, n_predicted_parameters)
