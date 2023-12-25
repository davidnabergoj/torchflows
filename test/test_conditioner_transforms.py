import pytest
import torch

from normalizing_flows.bijections.finite.autoregressive.conditioning.transforms import (
    MADE, FeedForward, LinearMADE, ResidualFeedForward, Linear, ConditionerTransform
)
from test.constants import __test_constants


@pytest.mark.parametrize('transform_class', [
    MADE,
    LinearMADE,
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('input_event_shape', __test_constants['input_event_shape'])
@pytest.mark.parametrize('output_event_shape', __test_constants['output_event_shape'])
@pytest.mark.parametrize('parameter_shape_per_element', __test_constants['parameter_shape_per_element'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
def test_autoregressive(transform_class,
                        batch_shape,
                        input_event_shape,
                        output_event_shape,
                        parameter_shape_per_element,
                        context_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *input_event_shape))
    transform: ConditionerTransform = transform_class(
        input_event_shape=input_event_shape,
        output_event_shape=output_event_shape,
        parameter_shape_per_element=parameter_shape_per_element,
        context_shape=context_shape,
    )

    if context_shape is not None:
        c = torch.randn(size=(*batch_shape, *context_shape))
        out = transform(x, c)
    else:
        out = transform(x)
    assert out.shape == (*batch_shape, *output_event_shape, *parameter_shape_per_element)


@pytest.mark.parametrize('transform_class', [
    FeedForward,
    ResidualFeedForward,
    Linear
])
@pytest.mark.parametrize('batch_shape', __test_constants['batch_shape'])
@pytest.mark.parametrize('input_event_shape', __test_constants['input_event_shape'])
@pytest.mark.parametrize('context_shape', __test_constants['context_shape'])
@pytest.mark.parametrize('predicted_parameter_shape', __test_constants['predicted_parameter_shape'])
def test_neural_network(transform_class,
                        batch_shape,
                        input_event_shape,
                        context_shape,
                        predicted_parameter_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *input_event_shape))
    transform: ConditionerTransform = transform_class(
        input_event_shape=input_event_shape,
        context_shape=context_shape,
        parameter_shape=predicted_parameter_shape
    )

    if context_shape is not None:
        c = torch.randn(size=(*batch_shape, *context_shape))
        out = transform(x, c)
    else:
        out = transform(x)
    assert out.shape == (*batch_shape, *predicted_parameter_shape)
